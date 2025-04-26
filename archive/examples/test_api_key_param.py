from dotenv import load_dotenv
import os
from langchain_asi import ChatASI
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
asi_api_key = os.getenv("ASI_API_KEY")

# Pass it to the constructor
llm = ChatASI(
    model_name="asi1-mini",
    temperature=0.7,
    asi_api_key=asi_api_key  # Passing the API key loaded from .env
)

# Test the model
response = llm.invoke([HumanMessage(content="Tell me a short joke about programming.")])
print("Response with explicit API key:")
print(response.content)

# For comparison, create another instance without explicitly passing the API key
llm2 = ChatASI(
    model_name="asi1-mini",
    temperature=0.7
    # API key will be loaded from environment variable
)

# Test the second model
response2 = llm2.invoke([HumanMessage(content="Tell me another short joke about programming.")])
print("\nResponse with environment variable API key:")
print(response2.content)
