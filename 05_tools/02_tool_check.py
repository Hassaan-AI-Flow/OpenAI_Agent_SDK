from agents import (
    Agent,
    Runner,
    ToolsToFinalOutputResult,
    function_tool,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
set_tracing_disabled(True)

# ---------------------
# Define Tools with Decorators
# ---------------------

@function_tool
async def check_price(product: str) -> float:
    return 99.99

@function_tool
async def check_stock(product: str) -> int:
    return 10

@function_tool
async def check_reviews(product: str) -> str:
    return "Great product, 4.5 stars!"

# ---------------------
# Fix: Custom Tool Use Behavior Function
# ---------------------

async def custom_tool_handler(context, tool_results) -> ToolsToFinalOutputResult:
    tool_result = tool_results[0]
    tool_name = tool_result.tool.name
    result = tool_result.output
    if tool_name == "check_price" and result > 50:
        return ToolsToFinalOutputResult(True, f"Price is high: ${result}")
    return ToolsToFinalOutputResult(False)


# ---------------------
# Set Up Client & Model
# ---------------------

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    openai_client=client,
    model="gemini-2.0-flash-exp",
)

# ---------------------
# Main Async Function
# ---------------------

async def main():
    agent_run_llm = Agent(
        name="Store Agent (Run LLM Again)",
        instructions="Answer customer queries about products.",
        tools=[check_price, check_stock, check_reviews],
        tool_use_behavior="run_llm_again",
        reset_tool_choice=True,
        model=model,
    )

    agent_stop_first = Agent(
        name="Store Agent (Stop on First Tool)",
        instructions="Answer customer queries about products.",
        tools=[check_price, check_stock, check_reviews],
        tool_use_behavior="stop_on_first_tool",
        reset_tool_choice=True,
        model=model,
    )

    agent_custom = Agent(
        name="Store Agent (Custom Function)",
        instructions="Answer customer queries about products.",
        tools=[check_price, check_stock, check_reviews],
        tool_use_behavior=custom_tool_handler,
        reset_tool_choice=True,
        model=model,
    )

    agent_no_reset = Agent(
        name="Store Agent (No Reset)",
        instructions="Answer customer queries about products.",
        tools=[check_price, check_stock, check_reviews],
        tool_use_behavior="run_llm_again",
        reset_tool_choice=False,
        model=model,
    )

    # ---------------------
    # Test 1: Price Query
    # ---------------------
    print("=== Price Query ===")
    result = await Runner.run(agent_run_llm, "What's the price of a phone?")
    print("Run LLM Again:", result.final_output)

    result = await Runner.run(agent_stop_first, "What's the price of a phone?")
    print("Stop on First Tool:", result.final_output)

    result = await Runner.run(agent_custom, "What's the price of a phone?")
    print("Custom Function:", result.final_output)

    # ---------------------
    # Test 2: Stock Query
    # ---------------------
    print("\n=== Stock Query ===")
    result = await Runner.run(agent_run_llm, "How many phones are in stock?")
    print("Run LLM Again:", result.final_output)

    result = await Runner.run(agent_stop_first, "How many phones are in stock?")
    print("Stop on First Tool:", result.final_output)

    result = await Runner.run(agent_custom, "How many phones are in stock?")
    print("Custom Function:", result.final_output)

    # ---------------------
    # Test 3: No Reset Tool Choice
    # ---------------------
    print("\n=== No Reset Tool Choice ===")
    result = await Runner.run(agent_no_reset, "What's the price of a phone?")
    print("First Run:", result.final_output)

    result = await Runner.run(agent_no_reset, "How many phones are in stock?")
    print("Second Run (No Reset):", result.final_output)

# ---------------------
# Run the Main Function
# ---------------------
asyncio.run(main())