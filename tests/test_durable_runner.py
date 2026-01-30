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


@pytest.mark.asyncio
async def test_tool_call(dbos_env):
    """DurableRunner executes a tool call and returns the final message."""
    tool_calls_made = []

    @function_tool
    def get_weather(city: str) -> str:
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


@pytest.mark.asyncio
async def test_multiple_tool_calls(dbos_env):
    """DurableRunner handles multiple parallel tool calls in deterministic order."""
    call_order = []

    @function_tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        call_order.append(city)
        return f"Sunny in {city}"

    model = FakeModel([
        # Model requests two tool calls at once
        ModelResponse(
            output=[
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call_1",
                    name="get_weather",
                    arguments='{"city": "NYC"}',
                ),
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call_2",
                    name="get_weather",
                    arguments='{"city": "LA"}',
                ),
            ],
            usage=Usage(),
            response_id="resp_1",
        ),
        make_message_response("NYC and LA are both sunny."),
    ])
    agent = Agent(name="test", model=model, tools=[get_weather])

    @DBOS.workflow()
    async def wf(user_input: str):
        result = await DurableRunner.run(agent, user_input)
        return result.final_output

    output = await wf("Weather in NYC and LA?")
    assert output == "NYC and LA are both sunny."
    # Turnstile ensures deterministic ordering matching the model response order
    assert call_order == ["NYC", "LA"]
