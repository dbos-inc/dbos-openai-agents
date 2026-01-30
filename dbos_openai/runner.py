import dataclasses
from asyncio import Event
from typing import Any, AsyncIterator, List

from agents import (
    Agent,
    Handoff,
    Model,
    RunConfig,
    Runner,
    RunResult,
    TContext,
)
from agents.items import ModelResponse, TResponseOutputItem, TResponseStreamEvent
from agents.models.multi_provider import MultiProvider
from agents.tool import FunctionTool, Tool
from agents.tool_context import ToolContext

from dbos import DBOS


# ---------------------------------------------------------------------------
# Turnstile: ordered execution of concurrent async operations
# ---------------------------------------------------------------------------

class Turnstile:
    """Serializes concurrent async operations in a fixed order by ID.

    When the OpenAI agents SDK fires multiple tool calls via asyncio.gather,
    DBOS needs them to hit their steps in a deterministic order so that
    function_id assignment is consistent on replay.
    """

    def __init__(self, ids: list[str]):
        self.turns = dict(zip(ids, ids[1:]))
        self.events = {id: Event() for id in ids}
        self.canceled = False
        if ids:
            self.events[ids[0]].set()

    async def wait_for(self, id: str) -> None:
        event = self.events[id]
        await event.wait()
        if self.canceled:
            raise _CanceledError()

    def allow_next_after(self, id: str) -> None:
        next_id = self.turns.get(id)
        if next_id is not None:
            self.events[next_id].set()

    def cancel_all_after(self, id: str) -> None:
        self.canceled = True
        next_id = self.turns.get(id)
        while next_id is not None:
            self.events[next_id].set()
            next_id = self.turns.get(next_id)


class _CanceledError(Exception):
    """Raised when a turnstile-gated operation is canceled."""


# ---------------------------------------------------------------------------
# Shared execution state
# ---------------------------------------------------------------------------

class _State:
    __slots__ = ("turnstile",)

    def __init__(self) -> None:
        self.turnstile = Turnstile([])


# ---------------------------------------------------------------------------
# DBOS step wrappers (module-level so they're registered at import time)
# ---------------------------------------------------------------------------

@DBOS.step(retries_allowed=True, max_attempts=10, interval_seconds=1.0, backoff_rate=2.0)
async def _model_call_step(call_fn):
    """Execute an LLM call as a durable DBOS step with retries."""
    return await call_fn()


@DBOS.step()
async def _tool_call_step(call_fn):
    """Execute a tool call as a durable DBOS step."""
    return await call_fn()


# ---------------------------------------------------------------------------
# Model provider / wrapper
# ---------------------------------------------------------------------------

def _get_function_call_ids(output: List[TResponseOutputItem]) -> List[str]:
    """Extract function call IDs from a model response."""
    return [item.call_id for item in output if item.type == "function_call"]


class DBOSModelProvider(MultiProvider):
    """Model provider that wraps every model in a DBOSModelWrapper."""

    def __init__(self, state: _State):
        super().__init__()
        self._state = state

    def get_model(self, model_name: str | None) -> Model:
        model = super().get_model(model_name or None)
        return DBOSModelWrapper(model, self._state)


class DBOSModelWrapper(Model):
    """Wraps a Model so each get_response() call is a durable DBOS step."""

    def __init__(self, model: Model, state: _State):
        self.model = model
        self.model_name = "DBOSModelWrapper"
        self._state = state

    async def get_response(self, *args, **kwargs) -> ModelResponse:
        async def call_llm():
            return await self.model.get_response(*args, **kwargs)

        result = await _model_call_step(call_llm)

        # Prepare the turnstile for any tool calls in the response
        ids = _get_function_call_ids(result.output)
        self._state.turnstile = Turnstile(ids)

        return result

    def stream_response(self, *args, **kwargs) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError(
            "Streaming is not supported in durable mode. Use DurableRunner.run() instead."
        )


# ---------------------------------------------------------------------------
# Tool wrapping
# ---------------------------------------------------------------------------

def _create_tool_wrapper(state: _State, tool: FunctionTool):
    """Create a turnstile-gated, DBOS-step-wrapped on_invoke_tool."""

    async def on_invoke_tool_wrapper(tool_context: ToolContext[Any], tool_input: str) -> Any:
        turnstile = state.turnstile
        call_id = tool_context.tool_call_id

        await turnstile.wait_for(call_id)
        try:
            async def call_tool():
                # Signal the next tool to start now that this step has been
                # entered and assigned its function_id.  The actual tool work
                # below can then run concurrently with subsequent tools.
                turnstile.allow_next_after(call_id)
                return await tool.on_invoke_tool(tool_context, tool_input)

            result = await _tool_call_step(call_tool)
            # Also signal here for the replay path where call_tool is never
            # invoked (DBOS returns the cached result instead).
            turnstile.allow_next_after(call_id)
            return result
        except BaseException as ex:
            turnstile.cancel_all_after(call_id)
            raise ex from None

    return on_invoke_tool_wrapper


def _wrap_agent(agent: Agent[TContext], state: _State) -> Agent[TContext]:
    """Return a clone of *agent* with model and tools wrapped for DBOS durability."""

    clone_kwargs: dict[str, Any] = {}

    # Wrap the model if it's a Model instance (the SDK uses it directly,
    # bypassing the model_provider — see agents/run.py:2043).
    if isinstance(agent.model, Model) and not isinstance(agent.model, DBOSModelWrapper):
        clone_kwargs["model"] = DBOSModelWrapper(agent.model, state)

    wrapped_tools: list[Tool] = []
    for tool in agent.tools:
        if isinstance(tool, FunctionTool):
            wrapper = _create_tool_wrapper(state, tool)
            wrapped_tools.append(dataclasses.replace(tool, on_invoke_tool=wrapper))
        else:
            wrapped_tools.append(tool)
    clone_kwargs["tools"] = wrapped_tools

    wrapped_handoffs: list[Agent[Any] | Handoff[Any]] = []
    for handoff in agent.handoffs:
        if isinstance(handoff, Agent):
            wrapped_handoffs.append(_wrap_agent(handoff, state))
        elif isinstance(handoff, Handoff):
            wrapped_handoffs.append(_wrap_handoff(handoff, state))
        else:
            raise TypeError(f"Unsupported handoff type: {type(handoff)}")
    clone_kwargs["handoffs"] = wrapped_handoffs

    return agent.clone(**clone_kwargs)


def _wrap_handoff(handoff: Handoff[TContext], state: _State) -> Handoff[TContext]:
    """Wrap a Handoff so the agent it produces also has wrapped tools."""
    original = handoff.on_invoke_handoff

    async def wrapped(*args, **kwargs) -> Any:
        agent = await original(*args, **kwargs)
        return _wrap_agent(agent, state)

    return dataclasses.replace(handoff, on_invoke_handoff=wrapped)


# ---------------------------------------------------------------------------
# DurableRunner — main entry point
# ---------------------------------------------------------------------------

class DurableRunner:
    """Run an OpenAI agent with DBOS durability.

    Must be called from within a ``@DBOS.workflow()`` to get durable
    execution.  When called outside a workflow, model and tool calls
    execute normally without durability.

    Example::

        @DBOS.workflow()
        async def run_agent(user_input: str):
            result = await DurableRunner.run(agent, user_input)
            return result.final_output
    """

    @staticmethod
    async def run(
        starting_agent: Agent[TContext],
        input: str | list,
        **kwargs,
    ) -> RunResult:
        state = _State()

        run_config = kwargs.pop("run_config", RunConfig())
        run_config = dataclasses.replace(
            run_config,
            model_provider=DBOSModelProvider(state),
        )

        agent = _wrap_agent(starting_agent, state)

        return await Runner.run(
            starting_agent=agent,
            input=input,
            run_config=run_config,
            **kwargs,
        )
