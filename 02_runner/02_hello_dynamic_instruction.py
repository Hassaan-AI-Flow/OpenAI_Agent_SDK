from agents import Agent , Runner , set_tracing_disabled , OpenAIChatCompletionsModel , RunContextWrapper
from openai import AsyncOpenAI
import os , asyncio
from dotenv import load_dotenv , find_dotenv
from pydantic import BaseModel
from typing import Any

load_dotenv(find_dotenv())
set_tracing_disabled(True)



client = AsyncOpenAI(           
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client = client,
    model = "gemini-2.0-flash-exp"
)

class Instructions(BaseModel):
    """Instructions for the agent. All fields are optional."""
    name :  str 
    city :  str 
    age :  int
    about : Any 
async def my_instructions(
    ctx: RunContextWrapper[Instructions],
    agent: Agent[Instructions]
) -> str:   
    name_field  = ctx.context.name 
    city_field  = ctx.context.city  
    age_field = ctx.context.age
    about_field = ctx.context.about
    return f"""
    Agent Name: {agent.name}
    Hello! I am here to assist you...
    You are a helpful assistant. The user's name is {name_field}, they are {age_field} years old and live in {city_field} and
    here is some info about them: {about_field}.
    Use this information to personalize your responses. Always refer to the user by their name when appropriate.
    """

async def main():
    agent = Agent(
        name = "Helpful Assistant",
        instructions = my_instructions,
        model = model,
    )
    
    name = input("Enter your name: ")
    city = input("Enter your city: ")
    age = int(input("Enter your age: "))
    about = input("Enter some info about yourself: ")
    while True:
        user_input = input("Tell me (or type 'exit' to quit): ")

        if user_input.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = await Runner.run(
            starting_agent=agent,
            input=user_input,
            context=Instructions(name=name, city=city, age=age,about=about),
            )

        print("Output :", result.final_output)

if __name__ == "__main__":
    print("\n[STARTING AGENT]\n")
    asyncio.run(main())