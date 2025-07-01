import os
import logging
from dotenv import load_dotenv
from agents import (Agent ,
                    Runner ,
                    set_tracing_disabled , 
                    OpenAIChatCompletionsModel , 
                    function_tool,
                    RunContextWrapper,
                    AgentHooks,
                    Tool,)
from openai import AsyncOpenAI
from typing import Any
load_dotenv()
set_tracing_disabled(True)
logging.getLogger("httpx").setLevel(logging.WARNING)


class CustomAgentHooks(AgentHooks):
    def __init__(self, display_name: str):
        self.event_counter = 0
        self.display_name = display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended with output {output}"
        )

    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source: Agent) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {source.name} handed off to {agent.name}"
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started tool {tool.name}"
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended tool {tool.name} with result {result}"
        )


###

@function_tool
async def currency_converter(amount: float, from_currency: str) -> str:
    """Convert the given amount between dollar and pkr. Use 'USD' for dollar and 'PKR' for rupees."""
    USD_TO_PKR = 283.68
    if from_currency.upper() == "USD":
        converted = amount * USD_TO_PKR
        return f"{amount} USD is {converted:.2f} PKR"
    elif from_currency.upper() == "PKR":
        converted = amount / USD_TO_PKR
        return f"{amount} PKR is {converted:.2f} USD"
    else:
        return "Please specify 'USD' or 'PKR' as the currency."
    
@function_tool
def simple_addition(x: int, y: int) -> str:
    """Adds two numbers and returns the result."""
    return f"The sum of {x} and {y} is {x + y}"

client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    openai_client = client,
    model="gemini-2.5-flash",
)

math_agent = Agent(
    name="Math_Agent",
    instructions=(
        "You are a math expert. You can solve math problems and provide step-by-step solutions.\n"
        "Here are some examples:\n"
        "Q: What is 12 + 7?\n"
        "A: Let's think step by step. 12 + 7 = 19. The answer is 19.\n"
        "Q: If a rectangle has length 5 and width 3, what is its area?\n"
        "A: Let's think step by step. Area = length × width = 5 × 3 = 15. The answer is 15.\n"
        "Q: What is the square root of 81?\n"
        "A: Let's think step by step. The square root of 81 is 9. The answer is 9."
    ),
    model=model,
    handoff_description="Solve math problems and provide step-by-step solutions.",
    hooks=CustomAgentHooks(display_name="Math_Query_Solver")
)

translate_agent = Agent(
    name = "Language Translator",
    instructions=(
        "You are a language translator. You can translate text from one language to another.\n"
        "Here are some examples:\n"
        "Q: Translate 'Hello, how are you?' to French.\n"
        "A: Let's think step by step. The phrase 'Hello, how are you?' in French is 'Bonjour, comment ça va ?'.\n"
        "Q: Translate 'Good morning' to Spanish.\n"
        "A: Let's think step by step. The phrase 'Good morning' in Spanish is 'Buenos días'.\n"
        "Q: Translate 'Thank you' to German.\n"
        "A: Let's think step by step. The phrase 'Thank you' in German is 'Danke'."
    ),
    model=model,
    handoff_description="Translate language from one language to another.",
    hooks=CustomAgentHooks(display_name="Language_Translator")
)

main_agent = Agent(
    name="Helpful_Assistant",
    instructions="""U are a helpful assistant u take the user queries and think step by step and
     give the answer to the user.u can use tools and handoffs agent to answer the user queries.""",
    model=model,
    tools = [currency_converter,simple_addition,],
    handoffs=[math_agent, translate_agent],
    hooks=CustomAgentHooks(display_name="source agent")
)

try:
    while True:
        print("\n---Welcome to the Helpful Assistant!---\n")
        user_input = input("Enter your question(Type 'exit' to quit): ")
        if user_input.lower().strip() == 'exit':
            break
        response = Runner.run_sync(main_agent,
                                   user_input,
                                    )
        print("\n" , response.final_output)
except Exception as e :
    print(f"An Error Occurred : {e}.")
