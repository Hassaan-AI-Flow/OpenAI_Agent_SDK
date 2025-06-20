from agents import Agent , Runner , set_tracing_disabled , OpenAIChatCompletionsModel
from openai import AsyncOpenAI
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

async def main():
    agent = Agent(
        name = "Helpful_Assistant",   #if this is not set, the agent will raise the error
        instructions="""You are a helpful assistant that provides information and answers 
        questions to the best of your ability.""", #if this is not set, the agent will run properly
        model=model,
    )
    response = await Runner.run(
        starting_agent=agent,
        input=input("tell me:")
    )
    print(response.final_output)

if __name__ == "__main__":
    asyncio.run(main())