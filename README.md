# Durable OpenAI Agents

Durable execution for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) using [DBOS](https://github.com/dbos-inc/dbos-transact-py).

## Installation

```bash
pip install dbos-openai-agents
```

## Usage

Call your agent using `DBOSRunner.run()` from a `@DBOS.workflow()`.
Annotate tool calls and guardrails with `@DBOS.step()`.

```python
import asyncio
from agents import Agent, function_tool
from dbos import DBOS, DBOSConfig
from dbos_openai import DBOSRunner

# Decorate tool calls and guardrails with @DBOS.step() for durable execution
@function_tool
@DBOS.step()
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"

agent = Agent(name="weather", tools=[get_weather])

# Use DBOSRunner to call your agent from a workflow
@DBOS.workflow()
async def run_agent(user_input: str) -> str:
    result = await DBOSRunner.run(agent, user_input)
    return str(result.final_output)


async def main():
    output = await run_agent("How is the weather in San Francisco")
    print(output)


if __name__ == "__main__":
    config: DBOSConfig = {
        "name": "my-agent",
    }
    DBOS(config=config)
    DBOS.launch()
    asyncio.run(main())
```

`DBOSRunner.run()` is a drop-in replacement for `Runner.run()` with the same arguments.
It must be called from within a `@DBOS.workflow()`.
