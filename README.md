# dbos-openai

Durable execution for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) using [DBOS](https://github.com/dbos-inc/dbos-transact-py).

## Installation

```bash
pip install dbos-openai
```

## Usage

Wrap your agent run in a DBOS workflow using `DurableRunner.run()`:

```python
from agents import Agent, function_tool
from dbos import DBOS
from dbos_openai import DurableRunner

# Decorate tool calls and guardrails with @DBOS.step() for durable execution
@function_tool
@DBOS.step()
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"

agent = Agent(name="weather", tools=[get_weather])

@DBOS.workflow()
async def run_agent(user_input: str) -> str:
    result = await DurableRunner.run(agent, user_input)
    return str(result.final_output)
```

`DurableRunner.run()` is a drop-in replacement for `Runner.run()` with the same arguments.
It must be called from within a `@DBOS.workflow()`.
