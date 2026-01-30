import asyncio

import pytest

from dbos import DBOS
from agents import Agent, Usage
from agents.items import ModelResponse
from agents.tool import function_tool
from openai.types.responses import ResponseFunctionToolCall

from utils import FakeModel, make_message_response, make_tool_call_response
from dbos_openai import DurableRunner


@pytest.mark.asyncio
async def test_simple_message(dbos_env):
    """DurableRunner returns a simple text response."""
    model = FakeModel([make_message_response("Hello!")])
    agent = Agent(name="test", model=model)

    @DBOS.workflow()
    async def wf(user_input: str):
        result = await DurableRunner.run(agent, user_input)
        return result.final_output

    output = await wf("Hi")
    assert output == "Hello!"

    # 1 workflow, with 1 model call step
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 1
    assert steps[0]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_tool_call(dbos_env):
    """DurableRunner executes a tool call and returns the final message."""
    tool_calls_made = []

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        tool_calls_made.append(city)
        return f"Sunny in {city}"

    model = FakeModel([
        make_tool_call_response("call_1", "get_weather", '{"city": "NYC"}'),
        make_message_response("The weather in NYC is sunny."),
    ])
    agent = Agent(name="test", model=model, tools=[get_weather])

    @DBOS.workflow()
    async def wf(user_input: str):
        result = await DurableRunner.run(agent, user_input)
        return result.final_output

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
async def test_multiple_tool_calls(dbos_env):
    """DurableRunner handles parallel tool calls that start in deterministic order."""
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

    model = FakeModel([
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
    ])
    agent = Agent(name="test", model=model, tools=[get_weather])

    @DBOS.workflow()
    async def wf(user_input: str):
        result = await DurableRunner.run(agent, user_input)
        return result.final_output

    output = await wf("Weather everywhere?")
    assert output == "Done."
    # Tools actually run concurrently (not sequentially)
    assert max_concurrent > 1, f"Expected concurrent execution, but max_concurrent={max_concurrent}"

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
