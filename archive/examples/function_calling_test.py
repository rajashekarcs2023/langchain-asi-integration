"""Test function calling with ASI and compare to expected behavior."""

import os
import sys
from dotenv import load_dotenv
from langchain_asi import ChatASI
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Define a simple tool
@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the weather for a specific location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The temperature unit to use. 'celsius' or 'fahrenheit'
    
    Returns:
        The current weather for the specified location.
    """
    # This is a mock function that would normally call a weather API
    return f"The weather in {location} is sunny and 22 degrees {unit}."

def test_function_calling():
    """Test function calling with ASI."""
    print("Testing function calling with ASI...")
    
    # Initialize the chat model
    chat = ChatASI(
        model_name="asi1-mini",  # Use the appropriate model
        temperature=0,
        verbose=True
    )
    
    # Bind the tool to the model
    chat_with_tools = chat.bind_tools(
        tools=[get_weather],
        tool_choice="any"  # Force the model to use a tool
    )
    
    # Create a message that should trigger tool use
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can get weather information. Always use the get_weather tool when asked about weather."},
        {"role": "user", "content": "What's the weather like in San Francisco? Use the get_weather tool to find out."}
    ]
    
    try:
        # Invoke the model with the messages
        response = chat_with_tools.invoke(messages)
        print("\nResponse:")
        print(f"Type: {type(response)}")
        print(f"Content: {response.content}")
        
        # Check if tool calls were made
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        print(f"\nTool calls: {len(tool_calls)}")
        for i, tool_call in enumerate(tool_calls):
            print(f"\nTool call {i+1}:")
            print(f"Type: {tool_call.get('type')}")
            print(f"Function: {tool_call.get('function', {}).get('name')}")
            print(f"Arguments: {tool_call.get('function', {}).get('arguments')}")
        
        return True
    except Exception as e:
        print(f"\nError: {e}")
        return False

if __name__ == "__main__":
    success = test_function_calling()
    sys.exit(0 if success else 1)
