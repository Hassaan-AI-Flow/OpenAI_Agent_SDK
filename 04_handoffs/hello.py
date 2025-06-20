import os , asyncio
from agents import Agent , Runner , OpenAIChatCompletionsModel , set_tracing_disabled , handoff ,RunContextWrapper
from openai import AsyncOpenAI
from dotenv import load_dotenv , find_dotenv
from pydantic import BaseModel
load_dotenv(find_dotenv())
set_tracing_disabled(True)

client = AsyncOpenAI(
    api_key = os.getenv("GOOGLE_API_KEY"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client = client,
    model = "gemini-2.0-flash-exp",
)


class BillingResponse(BaseModel):
    def __init__(self, selling_price: float, return_price: float):
        self.selling_price = selling_price
        self.return_price = return_price
        self.loss = self.selling_price - self.return_price  # Calculate loss here

    def __str__(self):
        return (f"Selling Price: ${self.selling_price}, "
                f"Return Price: ${self.return_price}, "
                f"Loss: ${self.loss} billion dollars")


# Example of how to use this in an agent
async def custom_billing_agent_handler(context: 'RunContextWrapper[None]', 
                                       input: str) -> str:
    """
    Custom handler for the billing agent.
    It calculates the loss based on selling and return prices.
    """

    # Example prices (you can parse 'input' if needed)
    selling_price = 100.0  # You can replace this with dynamic values from input
    return_price = 90.0    # You can replace this with dynamic values from input

    billing_response = BillingResponse(selling_price, return_price)
    
    return str(billing_response)

# Output when run:
# Selling Price: $100.0, Return Price: $90.0, Loss: $10.0 billion dollars

billing_agent = Agent(
    name = "Billing Agent",
    model = model,
    instructions = "You are a billing agent. Your task is to handle billing inquiries and provide information about billing processes.",
)

offer_and_discount_agent = Agent(
    name = "Offer and Discount Agent",
    model = model,
    instructions = "You are an offer discount agent. Your task is to provide information about available discounts and offers.",
)

product_recommendation_agent = Agent(
    name = "Product Recommendation Agent",
    model = model,
    instructions = "You are a product recommendation agent. Your task is to recommend products based on user preferences and needs.",
)

product_refund_agent = Agent(
    name = "Product Refund Agent",
    model = model,
    instructions = "You are a product refund agent. Your task is to assist users with product refund requests and provide information about the refund process.",
)

greeting_agent = Agent(
    name = "Greeting Agent",
    model = model,
    instructions = "You are a greeting agent. Your task is to greet users and provide a warm welcome.",
        handoffs=[
        handoff(agent=billing_agent,
                tool_description_override="Hand off billing_agent-related questions",
                input_type = BillingResponse,
                on_handoff=custom_billing_agent_handler,
                input_filter = BillingResponse),   
        handoff(agent=offer_and_discount_agent,
                tool_description_override="Hand off offer_and_discount_agent-related questions"),
        handoff(agent=product_recommendation_agent ,
                tool_description_override="Hand off product_recommendation_agent -related questions"),
        handoff(agent=product_refund_agent,
                tool_description_override="Hand off product_refund_agent-related questions"),
    ]
)

async def main():
    while True:
        user_input = input("Enter your question (exit for stop): ")
        if user_input.lower() == "exit":
            print("Exiting the program.")
            break
        
        result = await Runner.run(
            starting_agent=greeting_agent,
            input=user_input
        )
        print(result.final_output)

asyncio.run(main())    
