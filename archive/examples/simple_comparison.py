"""Simple demonstration of ChatASI compared to ChatOpenAI."""
import os
from langchain_asi import ChatASI
from langchain_core.messages import HumanMessage

# Set API key from environment variable
# For a real implementation, set this in your environment
# export ASI_API_KEY="your-asi-api-key"

# Example with ChatASI
print("\nExample with ChatASI:")
print("\nChatASI automatically selects the correct API endpoint based on the model name:")
print("- 'asi1-mini' → https://api.asi1.ai/v1")
print("- other models → https://api.asi.ai/v1 (default)")

# Test with ASI1 model
llm_asi1 = ChatASI(
    model_name="asi1-mini",
    # asi_api_key is loaded from ASI_API_KEY environment variable
    # asi_api_base is automatically selected based on the model name
)

# Test with non-ASI1 model
llm_asi = ChatASI(
    model_name="asi-standard",  # This is a non-ASI1 model
    # asi_api_key is loaded from ASI_API_KEY environment variable
    # asi_api_base is automatically selected based on the model name
)

# Function to demonstrate usage
def demonstrate_llm(llm, model_type):
    print(f"\nDemonstrating ChatASI with {model_type} model:")
    
    # Simple string input
    print("\nSimple string input:")
    try:
        response = llm.invoke("Tell me a joke about programming.")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Using messages
    print("\nUsing messages:")
    try:
        response = llm.invoke([HumanMessage(content="What's the difference between a programmer and a non-programmer?")])
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error: {e}")

# Only run the demonstration if API key is available
if os.getenv("ASI_API_KEY"):
    # Test ASI1 model
    print("\nTesting ASI1 model (asi1-mini):")
    print(f"API endpoint: {llm_asi1.asi_api_base}")
    demonstrate_llm(llm_asi1, "ASI1")
    
    # Test non-ASI1 model
    print("\nTesting non-ASI1 model (asi-standard):")
    print(f"API endpoint: {llm_asi.asi_api_base}")
    demonstrate_llm(llm_asi, "non-ASI1")
else:
    print("Skipping ChatASI example - ASI_API_KEY not set")
    print("Please set the ASI_API_KEY environment variable:")
    print("export ASI_API_KEY=\"your-api-key\"")

print("\nChatASI follows the same interface pattern as ChatOpenAI from LangChain.")
print("You can use them interchangeably in your LangChain applications.")
print("The API endpoint is automatically selected based on the model name.")
