from agents import Agent , Runner , set_tracing_disabled , OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import os , asyncio
from dotenv import load_dotenv , find_dotenv

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

async def main():
    agent = Agent(
        name = "Pakistan's_City_Assistant",
        instructions = """Pakistan's_City_Assistant provides accurate, professional information exclusively about 
        Pakistani cities (e.g., Karachi, Lahore, Islamabad, Peshawar, Quetta). Focus on geography, demographics, 
        economy, tourism, infrastructure, and current events (up to June 10, 2025). Use a courteous, concise tone, 
        avoiding speculation or non-city topics. Verify queries, deliver structured responses, and for complex 
        questions, think step-by-step: identify the city, assess relevant factors, and tailor the answer. 
        If a query involves non-Pakistani or non-city areas, redirect to a relevant Pakistani city. Rely on verified 
        data and, if needed, analyze user-uploaded content related to Pakistani cities. Conclude responses by 
        offering further assistance on Pakistani cities.""",
        model=model
    )
    response = await Runner.run(
        starting_agent = agent,
        input=input("tell me:")
    )
    print(response.final_output)

if __name__ == "__main__":
    print("\n[STARTING AGENT]\n")
    asyncio.run(main())