from typing import AsyncIterator

from agents import Usage
from agents.models.interface import Model
from agents.items import ModelResponse, TResponseStreamEvent
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)


def make_message_response(text: str, response_id: str = "resp_1") -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_1",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        usage=Usage(),
        response_id=response_id,
    )


def make_tool_call_response(
    call_id: str, tool_name: str, arguments: str, response_id: str = "resp_1"
) -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                call_id=call_id,
                name=tool_name,
                arguments=arguments,
            )
        ],
        usage=Usage(),
        response_id=response_id,
    )


class FakeModel(Model):
    """A model that returns a sequence of canned responses."""

    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.call_count = 0

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        *,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    ) -> ModelResponse:
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

    def stream_response(self, *args, **kwargs) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError
