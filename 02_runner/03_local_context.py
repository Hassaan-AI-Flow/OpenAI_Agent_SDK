# """We give the agent a class object and pass the function as a tool but not give the actual information to llm .
# In the run time llm use specific user information and get the answer means am not giving my context to llm 
# this is called local context"""

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
    name : str
    age : str
    location: str = "Earth"

@function_tool
async def get_user_information(ctx: RunContextWrapper[Info]) -> str: #Agent run but not know information.
    """return the user information like name, age, and location"""
    return f"The user {ctx.context.name} is {ctx.context.age} and lives in {ctx.context.location}."


async def main():
    context = Info(name="Hassaan", age=19 ,location="Pakistan",)
    agent = Agent[Info](
        name = "Helpful_Assistant",   
        instructions="""[You are a helpful assistant that provides information and answers 
        questions to the best of your ability. , Your specialization in mathematics]""",
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
            context =context,  
        )
        print(response.final_output)

if __name__ == "__main__":
    asyncio.run(main())
