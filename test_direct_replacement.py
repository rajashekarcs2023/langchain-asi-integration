import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_asi import ChatASI

# Load environment variables
load_dotenv()

# Initialize the model
asi_llm = ChatASI()

# Test with a simple query - correct way with a message
response = asi_llm.invoke([HumanMessage(content="What is the capital of France?")])
print("Response with message:", response.content)

# Try with a string to see if it works (not standard LangChain usage)
try:
    response_string = asi_llm.invoke("What is the capital of France?")
    print("Response with string:", response_string.content)
    print("String input works")
except Exception as e:
    print(f"String input error: {e}")
