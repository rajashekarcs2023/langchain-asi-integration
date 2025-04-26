"""
Bind Tools Test for ChatASI

This script tests the bind_tools method of the ChatASI class
to ensure it properly supports tool binding functionality.
"""

import os
import asyncio
import json
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Define a simple Pydantic model for a tool
class GetWeather(BaseModel):
    """Get the current weather in a given location."""
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

# Define a simple function-based tool
@tool
def get_population(location: str) -> str:
    """Get the population of a given location."""
    # This is a mock implementation
    populations = {
        "new york": "8.4 million",
        "los angeles": "4 million",
        "chicago": "2.7 million",
        "san francisco": "815 thousand"
    }
    location_lower = location.lower()
    if location_lower in populations:
        return f"The population of {location} is approximately {populations[location_lower]}."
    return f"I don't have population data for {location}."

async def test_bind_tools():
    """Test the bind_tools method of the ChatASI class."""
    print("=" * 80)
    print("BIND TOOLS TEST")
    print("=" * 80)
    
    # Initialize the ChatASI model
    asi_chat = ChatASI(model_name="asi1-mini", verbose=True)
    
    # Test 1: Bind schema-based tools
    print("\n[Test 1: Bind schema-based tools]")
    try:
        # Bind the GetWeather tool
        model_with_tools = asi_chat.bind_tools([GetWeather])
        
        # Print the model configuration
        print(f"Model with tools: {model_with_tools}")
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant that can provide weather information."),
            HumanMessage(content="What's the weather like in San Francisco?")
        ]
        
        # Use the model with tools
        response = await model_with_tools.ainvoke(messages)
        print(f"Response: {response}")
        
        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            print("\nTool calls:")
            for tool_call in response.tool_calls:
                print(f"  Tool: {tool_call.get('name')}")
                print(f"  Arguments: {tool_call.get('args')}")
        elif hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
            print("\nTool calls (from additional_kwargs):")
            for tool_call in response.additional_kwargs["tool_calls"]:
                print(f"  Tool: {tool_call.get('function', {}).get('name')}")
                print(f"  Arguments: {tool_call.get('function', {}).get('arguments')}")
        else:
            print("\nNo tool calls were made.")
            print(f"Response content: {response.content}")
    except Exception as e:
        print(f"Error in bind schema-based tools test: {e}")
    
    # Test 2: Bind function-based tools
    print("\n[Test 2: Bind function-based tools]")
    try:
        # Bind the get_population tool
        model_with_function = asi_chat.bind_tools([get_population])
        
        # Print the model configuration
        print(f"Model with function: {model_with_function}")
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant that can provide population information."),
            HumanMessage(content="What's the population of New York?")
        ]
        
        # Use the model with function
        response = await model_with_function.ainvoke(messages)
        print(f"Response: {response}")
        
        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            print("\nTool calls:")
            for tool_call in response.tool_calls:
                print(f"  Tool: {tool_call.get('name')}")
                print(f"  Arguments: {tool_call.get('args')}")
        elif hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
            print("\nTool calls (from additional_kwargs):")
            for tool_call in response.additional_kwargs["tool_calls"]:
                print(f"  Tool: {tool_call.get('function', {}).get('name')}")
                print(f"  Arguments: {tool_call.get('function', {}).get('arguments')}")
        else:
            print("\nNo tool calls were made.")
            print(f"Response content: {response.content}")
    except Exception as e:
        print(f"Error in bind function-based tools test: {e}")
    
    # Test 3: Direct API call with tools
    print("\n[Test 3: Direct API call with tools]")
    try:
        # Define a tool schema directly
        weather_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                    },
                    "required": ["location"]
                }
            }
        }
        
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can provide weather information."},
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ]
        
        # Make a direct API call with tools
        response = await asi_chat.acompletion_with_retry(
            messages=messages,
            tools=[weather_tool],
            tool_choice="auto"
        )
        
        print(f"Direct API response: {json.dumps(response, indent=2)}")
        
        # Check for tool calls
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "tool_calls" in choice.get("message", {}):
                tool_calls = choice["message"]["tool_calls"]
                print("\nTool calls:")
                for tool_call in tool_calls:
                    print(f"  Tool: {tool_call.get('function', {}).get('name')}")
                    print(f"  Arguments: {tool_call.get('function', {}).get('arguments')}")
            else:
                print("\nNo tool calls were made.")
                print(f"Message content: {choice.get('message', {}).get('content')}")
    except Exception as e:
        print(f"Error in direct API call with tools test: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_bind_tools())
