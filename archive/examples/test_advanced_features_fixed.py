"""
Test Advanced Features (Fixed)

This script tests if our changes to the ChatASI implementation have fixed the issues
with advanced features like bind_tools and with_structured_output.
"""

import os
import asyncio
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool, tool

# Load environment variables
load_dotenv()

# Define a simple Pydantic model for structured output
class WeatherInfo(BaseModel):
    """Information about the weather in a location."""
    
    location: str = Field(description="The location for the weather information")
    temperature: float = Field(description="The temperature in Celsius")
    conditions: str = Field(description="The weather conditions (e.g., sunny, rainy)")
    humidity: Optional[float] = Field(None, description="The humidity percentage")

# Define a simple tool using the @tool decorator
@tool
def get_weather(location: str) -> Dict[str, Any]:
    """Get the current weather in a given location."""
    # This is a mock implementation
    return {
        "location": location,
        "temperature": 22.5,
        "conditions": "Sunny",
        "humidity": 65.0
    }

async def test_advanced_features():
    """Test if our changes have fixed the issues with advanced features."""
    print("=" * 80)
    print("TESTING ADVANCED FEATURES (FIXED)")
    print("=" * 80)
    
    # Initialize the ChatASI model
    asi_chat = ChatASI(model_name="asi1-mini", verbose=True)
    
    # Test 1: bind_tools
    print("\n[Test 1: bind_tools]")
    try:
        # Bind tools to the model
        model_with_tools = asi_chat.bind_tools([get_weather])
        
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
        print(f"Error in bind_tools test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: with_structured_output
    print("\n[Test 2: with_structured_output]")
    try:
        # Create a model with structured output
        structured_model = asi_chat.with_structured_output(WeatherInfo)
        
        # Print the model configuration
        print(f"Structured model: {structured_model}")
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant that can provide weather information."),
            HumanMessage(content="What's the weather like in Paris? Use a temperature of 20°C, sunny conditions, and 60% humidity.")
        ]
        
        # Use the model with structured output
        response = await structured_model.ainvoke(messages)
        print(f"Structured response: {response}")
        
        # Test with include_raw=True
        structured_model_with_raw = asi_chat.with_structured_output(WeatherInfo, include_raw=True)
        response_with_raw = await structured_model_with_raw.ainvoke(messages)
        print("\nStructured response with raw:")
        print(f"  Raw: {response_with_raw.get('raw')}")
        print(f"  Parsed: {response_with_raw.get('parsed')}")
        print(f"  Parsing error: {response_with_raw.get('parsing_error')}")
    except Exception as e:
        print(f"Error in with_structured_output test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: ainvoke with different input formats
    print("\n[Test 3: ainvoke with different input formats]")
    try:
        # Test with a list of messages (standard format)
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?")
        ]
        response = await asi_chat.ainvoke(messages)
        print(f"List of messages response: {response.content}")
        
        # Test with a single message
        message = HumanMessage(content="What is the capital of Italy?")
        response = await asi_chat.ainvoke(message)
        print(f"Single message response: {response.content}")
        
        # Test with a tuple (content, message) format
        try:
            response = await asi_chat.ainvoke(("content", "What is the capital of Germany?"))
            print(f"Tuple (content, message) response: {response.content}")
        except Exception as e:
            print(f"Tuple (content, message) error: {e}")
        
        # Test with a tuple (role, content) format
        try:
            response = await asi_chat.ainvoke(("human", "What is the capital of Spain?"))
            print(f"Tuple (role, content) response: {response.content}")
        except Exception as e:
            print(f"Tuple (role, content) error: {e}")
    except Exception as e:
        print(f"Error in ainvoke test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_advanced_features())
