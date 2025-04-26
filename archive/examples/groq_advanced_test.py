"""
Groq Advanced Features Test

This script tests advanced features of the ChatGroq class to understand how it
handles bind_tools, with_structured_output, and ainvoke methods.
"""

import os
import asyncio
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Define a simple Pydantic model for structured output
class WeatherInfo(BaseModel):
    """Information about the weather in a location."""
    
    location: str = Field(description="The location for the weather information")
    temperature: float = Field(description="The temperature in Celsius")
    conditions: str = Field(description="The weather conditions (e.g., sunny, rainy)")
    humidity: Optional[float] = Field(None, description="The humidity percentage")

# Define a simple tool
class GetWeather(BaseModel):
    """Get the current weather in a given location."""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

async def test_groq_advanced_features():
    """Test advanced features of the ChatGroq class."""
    print("=" * 80)
    print("GROQ ADVANCED FEATURES TEST")
    print("=" * 80)
    
    # Initialize the ChatGroq model
    groq_chat = ChatGroq(model="llama-3.1-8b-instant", verbose=True)
    
    # Test 1: bind_tools
    print("\n[Test 1: bind_tools]")
    try:
        # Bind tools to the model
        model_with_tools = groq_chat.bind_tools([GetWeather])
        
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
    
    # Test 2: with_structured_output
    print("\n[Test 2: with_structured_output]")
    try:
        # Create a model with structured output
        structured_model = groq_chat.with_structured_output(WeatherInfo)
        
        # Print the model configuration
        print(f"Structured model: {structured_model}")
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant that can provide weather information."),
            HumanMessage(content="What's the weather like in Paris?")
        ]
        
        # Use the model with structured output
        response = await structured_model.ainvoke(messages)
        print(f"Structured response: {response}")
        
        # Test with include_raw=True
        structured_model_with_raw = groq_chat.with_structured_output(WeatherInfo, include_raw=True)
        response_with_raw = await structured_model_with_raw.ainvoke(messages)
        print("\nStructured response with raw:")
        print(f"  Raw: {response_with_raw.get('raw')}")
        print(f"  Parsed: {response_with_raw.get('parsed')}")
        print(f"  Parsing error: {response_with_raw.get('parsing_error')}")
    except Exception as e:
        print(f"Error in with_structured_output test: {e}")
    
    # Test 3: ainvoke with different input formats
    print("\n[Test 3: ainvoke with different input formats]")
    try:
        # Test with a list of messages (standard format)
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?")
        ]
        response = await groq_chat.ainvoke(messages)
        print(f"List of messages response: {response.content}")
        
        # Test with a single message
        message = HumanMessage(content="What is the capital of Italy?")
        response = await groq_chat.ainvoke(message)
        print(f"Single message response: {response.content}")
        
        # Test with a string
        try:
            response = await groq_chat.ainvoke("What is the capital of Germany?")
            print(f"String response: {response.content}")
        except Exception as e:
            print(f"String input error: {e}")
    except Exception as e:
        print(f"Error in ainvoke test: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_groq_advanced_features())
