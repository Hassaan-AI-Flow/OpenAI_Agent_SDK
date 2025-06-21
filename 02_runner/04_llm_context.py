from agents import (Agent ,
                    Runner , 
                    set_tracing_disabled ,
                    OpenAIChatCompletionsModel ,
                    RunContextWrapper,
                    function_tool)
from openai import AsyncOpenAI
from pydantic import BaseModel
from dataclasses import dataclass
import os , asyncio
from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv())
set_tracing_disabled(True)

client = AsyncOpenAI(
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

model = OpenAIChatCompletionsModel(
    openai_client = client,
    model = "gemini-2.0-flash-exp",
    )

@dataclass
class Info:
    name: str
    uid: int

@function_tool
async def get_user_information(ctx: RunContextWrapper[Info]) -> str: 
    """greet the user with their name
    Args:
        greeting : A specialized greeting message for the user"""
    name = ctx.context.name
    return f"Hello {name}, how can I assist you today?"


async def main():
    context = Info(name="Hassaan", uid=1268)
    agent = Agent[Info](
        name = "Helpful_Assistant", 
        instructions="Always greet the user using <function call>get_user_information</function call> and welcome them to panaversity.",
        model=model,
        tools=[get_user_information],
        )
    while True:
        user_input = input("Tell me (or type 'exit' to quit): ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        response = await Runner.run(
            starting_agent=agent,
            input= user_input,  
            context =context,              )
        print(response.final_output)

if __name__ == "__main__":
    asyncio.run(main())
