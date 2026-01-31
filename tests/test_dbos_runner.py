import asyncio

import pytest
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    Usage,
    handoff,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
)
from agents.items import ModelResponse
from agents.tool import function_tool
from agents.tool_guardrails import ToolInputGuardrailData, ToolOutputGuardrailData
from dbos import DBOS
from openai.types.responses import ResponseFunctionToolCall
from utils import FakeModel, make_message_response, make_tool_call_response

from dbos_openai_agents import DBOSRunner


@pytest.mark.asyncio
async def test_simple_message(dbos_env: None) -> None:
    """DBOSRunner returns a simple text response."""
    model = FakeModel([make_message_response("Hello!")])
    agent = Agent(name="test", model=model)

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    output = await wf("Hi")
    assert output == "Hello!"

    # 1 workflow, with 1 model call step
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 1
    assert steps[0]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_tool_call(dbos_env: None) -> None:
    """DBOSRunner executes a tool call and returns the final message."""
    tool_calls_made: list[str] = []

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        tool_calls_made.append(city)
        return f"Sunny in {city}"

    model = FakeModel(
        [
            make_tool_call_response("call_1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    agent = Agent(name="test", model=model, tools=[get_weather])

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    output = await wf("What's the weather in NYC?")
    assert output == "The weather in NYC is sunny."
    assert tool_calls_made == ["NYC"]

    # 1 workflow, with 3 steps: model call, tool call, model call
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 3
    assert steps[0]["function_name"] == "_model_call_step"
    assert "get_weather" in steps[1]["function_name"]
    assert steps[2]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_multiple_tool_calls(dbos_env: None) -> None:
    """DBOSRunner handles parallel tool calls that start in deterministic order."""
    num_calls = 100
    cities = [f"city_{i}" for i in range(num_calls)]
    concurrent = 0
    max_concurrent = 0

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(1)
        concurrent -= 1
        return f"Sunny in {city}"

    model = FakeModel(
        [
            ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id=f"call_{i}",
                        name="get_weather",
                        arguments=f'{{"city": "{city}"}}',
                    )
                    for i, city in enumerate(cities)
                ],
                usage=Usage(),
                response_id="resp_1",
            ),
            make_message_response("Done."),
        ]
    )
    agent = Agent(name="test", model=model, tools=[get_weather])

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    output = await wf("Weather everywhere?")
    assert output == "Done."
    # Tools actually run concurrently (not sequentially)
    assert (
        max_concurrent > 1
    ), f"Expected concurrent execution, but max_concurrent={max_concurrent}"

    # 1 workflow, with 102 steps: 1 model call + 100 tool calls + 1 model call
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == num_calls + 2
    assert steps[0]["function_name"] == "_model_call_step"
    # Steps are ordered by function_id — verify each tool step recorded
    # the correct city output in deterministic order
    for i in range(num_calls):
        assert "get_weather" in steps[i + 1]["function_name"]
        assert steps[i + 1]["output"] == f"Sunny in {cities[i]}"
    assert steps[num_calls + 1]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_guardrails(dbos_env: None) -> None:
    """DBOSRunner works with DBOS step-annotated guardrails on tools and agent output."""

    @tool_input_guardrail
    @DBOS.step()
    async def validate_tool_input(
        data: ToolInputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        """Check tool input is acceptable."""
        return ToolGuardrailFunctionOutput.allow(output_info="input_ok")

    @tool_output_guardrail
    @DBOS.step()
    async def validate_tool_output(
        data: ToolOutputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        """Check tool output is acceptable."""
        return ToolGuardrailFunctionOutput.allow(output_info="output_ok")

    @function_tool(
        tool_input_guardrails=[validate_tool_input],
        tool_output_guardrails=[validate_tool_output],
    )
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    @output_guardrail
    @DBOS.step()
    async def check_output(
        context: RunContextWrapper,
        agent: Agent,
        output: str,
    ) -> GuardrailFunctionOutput:
        """Verify the output is not empty."""
        return GuardrailFunctionOutput(
            output_info={"length": len(output)},
            tripwire_triggered=len(output) == 0,
        )

    model = FakeModel(
        [
            make_tool_call_response("call_1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    agent = Agent(
        name="test",
        model=model,
        tools=[get_weather],
        output_guardrails=[check_output],
    )

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    output = await wf("What's the weather in NYC?")
    assert output == "The weather in NYC is sunny."

    # 1 workflow with 6 steps:
    #   model call, tool input guardrail, tool call, tool output guardrail,
    #   model call, output guardrail
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 6
    assert steps[0]["function_name"] == "_model_call_step"
    assert "validate_tool_input" in steps[1]["function_name"]
    assert "get_weather" in steps[2]["function_name"]
    assert "validate_tool_output" in steps[3]["function_name"]
    assert steps[4]["function_name"] == "_model_call_step"
    assert "check_output" in steps[5]["function_name"]


@pytest.mark.asyncio
async def test_handoff(dbos_env: None) -> None:
    """DBOSRunner handles agent handoffs between multiple agents."""

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    # Weather agent: handles weather queries via a tool
    weather_model = FakeModel(
        [
            make_tool_call_response("call_w1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    weather_agent = Agent(
        name="weather_agent", model=weather_model, tools=[get_weather]
    )

    # Router agent: hands off to the weather agent
    router_model = FakeModel(
        [
            make_tool_call_response("call_h1", "transfer_to_weather_agent", "{}"),
        ]
    )
    router_agent = Agent(
        name="router",
        model=router_model,
        handoffs=[weather_agent],
    )

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(router_agent, user_input)
        return str(result.final_output)

    output = await wf("What's the weather in NYC?")
    assert output == "The weather in NYC is sunny."

    # 1 workflow with 4 steps:
    #   model call (router → handoff), model call (weather → tool call),
    #   tool call (get_weather), model call (weather → message)
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 4
    assert steps[0]["function_name"] == "_model_call_step"
    assert steps[1]["function_name"] == "_model_call_step"
    assert "get_weather" in steps[2]["function_name"]
    assert steps[3]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_tool_failure(dbos_env: None) -> None:
    """When a parallel tool call fails, the SDK sends the error back to the model."""

    @function_tool
    @DBOS.step()
    async def good_tool(city: str) -> str:
        """A tool that succeeds."""
        return f"Result for {city}"

    @function_tool
    @DBOS.step()
    async def bad_tool(city: str) -> str:
        """A tool that always fails."""
        raise ValueError("Something went wrong")

    model = FakeModel(
        [
            ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="call_1",
                        name="good_tool",
                        arguments='{"city": "NYC"}',
                    ),
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="call_2",
                        name="bad_tool",
                        arguments='{"city": "LA"}',
                    ),
                ],
                usage=Usage(),
                response_id="resp_1",
            ),
            make_message_response("Handled the error."),
        ]
    )
    agent = Agent(name="test", model=model, tools=[good_tool, bad_tool])

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    output = await wf("Do things")
    assert output == "Handled the error."

    # 1 workflow with 4 steps:
    #   model call (returns 2 tool calls), good_tool, bad_tool (error),
    #   model call (returns final message)
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 4
    assert steps[0]["function_name"] == "_model_call_step"
    assert "good_tool" in steps[1]["function_name"]
    assert steps[1]["output"] == "Result for NYC"
    assert steps[1]["error"] is None
    assert "bad_tool" in steps[2]["function_name"]
    assert steps[2]["output"] is None
    assert "Something went wrong" in str(steps[2]["error"])
    assert steps[3]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_explicit_handoff(dbos_env: None) -> None:
    """DBOSRunner handles explicit Handoff objects (not raw Agent)."""

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    weather_model = FakeModel(
        [
            make_tool_call_response("call_w1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    weather_agent = Agent(
        name="weather_agent", model=weather_model, tools=[get_weather]
    )

    router_model = FakeModel(
        [
            make_tool_call_response("call_h1", "transfer_to_weather_agent", "{}"),
        ]
    )
    router_agent = Agent(
        name="router",
        model=router_model,
        handoffs=[handoff(weather_agent)],
    )

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(router_agent, user_input)
        return str(result.final_output)

    output = await wf("What's the weather in NYC?")
    assert output == "The weather in NYC is sunny."

    # 1 workflow with 4 steps:
    #   model call (router → handoff), model call (weather → tool call),
    #   tool call (get_weather), model call (weather → message)
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 4
    assert steps[0]["function_name"] == "_model_call_step"
    assert steps[1]["function_name"] == "_model_call_step"
    assert "get_weather" in steps[2]["function_name"]
    assert steps[3]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_replay(dbos_env: None) -> None:
    """Forking a completed workflow replays all steps from recorded outputs."""
    call_count = 0

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        nonlocal call_count
        call_count += 1
        return f"Sunny in {city}"

    model = FakeModel(
        [
            make_tool_call_response("call_1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    agent = Agent(name="test", model=model, tools=[get_weather])

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    # Run the workflow for the first time
    output = await wf("What's the weather in NYC?")
    assert output == "The weather in NYC is sunny."
    assert call_count == 1

    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    original_id = workflows[0].workflow_id
    steps = await DBOS.list_workflow_steps_async(original_id)
    assert len(steps) == 3

    # Fork from past the last step so all steps replay from recorded outputs.
    # function_ids are 1-based, so start_step must exceed the max function_id.
    max_function_id = steps[-1]["function_id"]
    handle = await DBOS.fork_workflow_async(original_id, max_function_id + 1)
    replay_output = await handle.get_result()
    assert replay_output == "The weather in NYC is sunny."

    # The tool was NOT re-executed during replay
    assert call_count == 1


def test_streaming_not_supported() -> None:
    """DBOSModelWrapper.stream_response raises NotImplementedError."""
    from dbos_openai_agents.runner import DBOSModelWrapper, _State

    wrapper = DBOSModelWrapper(FakeModel([]), _State())
    with pytest.raises(NotImplementedError, match="Streaming is not supported"):
        wrapper.stream_response()


@pytest.mark.asyncio
async def test_string_model_name(dbos_env: None) -> None:
    """When agent.model is a string, DBOSModelProvider resolves and wraps it."""
    from unittest.mock import patch

    from agents.models.multi_provider import MultiProvider

    fake = FakeModel([make_message_response("Hello!")])
    agent = Agent(name="test", model="fake-model")

    with patch.object(MultiProvider, "get_model", return_value=fake):

        @DBOS.workflow()
        async def wf(user_input: str) -> str:
            result = await DBOSRunner.run(agent, user_input)
            return str(result.final_output)

        output = await wf("Hi")

    assert output == "Hello!"

    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 1
    assert steps[0]["function_name"] == "_model_call_step"
