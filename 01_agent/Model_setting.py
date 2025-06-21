from agents import (Agent ,
                    Runner , 
                    set_tracing_disabled ,
                    OpenAIChatCompletionsModel ,
                    ModelSettings ,
                    function_tool,)
from openai import AsyncOpenAI
import os , asyncio
from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv())
set_tracing_disabled(True)

@function_tool
async def weather(city: str) -> str:
    """
    Get the current weather for a given city.
    """
    # Simulate a weather API call
    return f"The current weather in {city} is sunny with a temperature of 25°C."

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
        instructions="""[You are a helpful assistant that provides information and answers 
        questions to the best of your ability. , Your specialization in mathematics]""", #if this is not set, the agent will run properly
        model=model,
        model_settings=ModelSettings(
            temperature=2.0,  
            top_p=0.9, 
            max_tokens=500,  
            # Removed frequency_penalty and presence_penalty (not supported by Gemini)
        ),
        tools=[weather], 
        tool_use_behavior="auto",
    )

    response = await Runner.run(
        starting_agent=agent,
        input=input("tell me:"), 
        max_turns=1 
    )
    print(response.final_output)

if __name__ == "__main__":
    asyncio.run(main())