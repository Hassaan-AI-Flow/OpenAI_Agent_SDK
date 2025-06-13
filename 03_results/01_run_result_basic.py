from agents import Agent , Runner , set_tracing_disabled , OpenAIChatCompletionsModel ,RunResult
from openai import AsyncOpenAI
import os , asyncio
from dotenv import load_dotenv , find_dotenv

load_dotenv(find_dotenv())
set_tracing_disabled(True)



client = AsyncOpenAI(
    api_key = os.getenv("GOOGLE_API_KEY"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client = client,
    model = "gemini-2.0-flash-exp"
)

agent = Agent(
    name = "EchoAgent",
    instructions = """You are an Echo Agent. Repeat the user's input prefixed with 'Echo: '.""",
    model = model,
)

async def main():
    print("--- Running 01_run_result_basics.py ---")

    user_input = "Hello, Agent!"
    print(f"\nOriginal User Input : '{user_input}'")

    try:
        result = await Runner.run(
            starting_agent=agent,
            input=user_input,
        )
        # 1. Final Output
        print(f"\n 1. Final Output:'{result.final_output}' (Type: {type(result.final_output)})")

        # 2. Last Agent
        print(f"\n 2. Last Agent: Name = '{result.last_agent.name}'(Type: {type(result.final_output)})")

        # 3. New Items (items generated during this run)
        print(f"\n 3. New Items: (Count: {len(result.new_items)})")
        if result.new_items:
            for i, item in enumerate(result.new_items):
                print(f"   - Item {i+1}: {str(item)[:100]}... (Type: {type(item)})")
        else:
            print("   No new items generated in this simple run (besides final_output which is part of it).")

        # 4. Original Input
        print(f"\n 4. Original User Input: '{result.input}' (Type: {type(result.input)})")


        print(f"\n 5. Input List for Next Turn (from to_input_list(), Count: {len(result.to_input_list())}):")
        for i, item in enumerate(result.to_input_list()):
        #     # to_input_list() returns a list of dicts or SDK items suitable for OpenAI API
            print(f"   - Item {i+1}: {str(item)[:100]}... (Type: {type(item)})")



    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

    print("\n--- Finished 01_run_result_basics.py ---")
if __name__ == "__main__":
    print("\n[STARTING AGENT]\n")
    asyncio.run(main())