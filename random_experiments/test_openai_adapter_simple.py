"""
Simple test for the OpenAI-compatible ASI adapter.
"""

import os
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_asi.openai_adapter import create_openai_compatible_model

# Load environment variables
load_dotenv()

def test_openai_adapter_simple():
    """Test the OpenAI-compatible ASI adapter with a simple query."""
    # Create a ChatASI model first
    asi_model = ChatASI(
        model_name="asi1-mini",
        asi_api_key=os.getenv("ASI_API_KEY"),
        asi_api_base=os.getenv("ASI_API_BASE", "https://api.asi1.ai/v1"),
    )
    
    # Create an OpenAI-compatible model
    model = create_openai_compatible_model(model=asi_model)
    
    # Make a simple query
    response = model.invoke("What is quantum computing?")
    
    # Print the response
    print("\nResponse from OpenAI-compatible ASI model:")
    print(response.content)
    
    # Test with a tool
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "description": "The temperature unit to use: 'celsius' or 'fahrenheit'",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    # Create a model with the tool
    model_with_tool = model.bind_tools(tools=[weather_tool])
    
    # Make a query that should use the tool
    response = model_with_tool.invoke("What's the weather like in New York?")
    
    # Print the response and tool calls
    print("\nResponse from OpenAI-compatible ASI model with tool:")
    print(response.content)
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        print("\nTool calls:")
        for tool_call in response.tool_calls:
            print(f"  Tool: {tool_call.get('name')}")
            print(f"  Arguments: {tool_call.get('args')}")
    elif hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
        print("\nTool calls from additional_kwargs:")
        for tool_call in response.additional_kwargs["tool_calls"]:
            if "function" in tool_call:
                print(f"  Tool: {tool_call['function'].get('name')}")
                print(f"  Arguments: {tool_call['function'].get('arguments')}")
    else:
        print("\nNo tool calls detected.")

if __name__ == "__main__":
    test_openai_adapter_simple()
