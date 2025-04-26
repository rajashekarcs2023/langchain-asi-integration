"""
Advanced Features Test for ChatASI

This script tests advanced features of the ChatASI class:
1. bind_tools - For binding tools to the chat model
2. with_structured_output - For returning outputs formatted to match a given schema
3. ainvoke - For asynchronous invocation
"""

import os
import asyncio
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Define a Pydantic model for structured output
class WeatherInfo(BaseModel):
    """Information about the weather in a location."""
    
    location: str = Field(description="The location for the weather information")
    temperature: float = Field(description="The temperature in Celsius")
    conditions: str = Field(description="The weather conditions (e.g., sunny, rainy)")
    humidity: Optional[float] = Field(None, description="The humidity percentage")

# Define a tool for getting weather
class GetWeather(BaseModel):
    """Get the current weather in a given location."""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

# Define a tool for getting population
class GetPopulation(BaseModel):
    """Get the current population in a given location."""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

async def test_advanced_features():
    """Test advanced features of the ChatASI class."""
    print("=" * 80)
    print("ADVANCED FEATURES TEST")
    print("=" * 80)
    
    # Initialize the ChatASI model
    asi_chat = ChatASI(model_name="asi1-mini", verbose=True)
    
    # Test 1: bind_tools
    print("\n[Test 1: bind_tools]")
    try:
        # Bind tools to the model
        model_with_tools = asi_chat.bind_tools([GetWeather, GetPopulation])
        
        # Test the model with a query that should trigger tool calling
        # Use a HumanMessage instead of a string
        response = await model_with_tools.ainvoke([HumanMessage(content="What is the population of New York?")])
        print(f"Response: {response.content}")
        
        # Check if tool calls were made
        if hasattr(response, "tool_calls") and response.tool_calls:
            print("\nTool calls:")
            for tool_call in response.tool_calls:
                print(f"  Tool: {tool_call.get('name')}")
                print(f"  Arguments: {tool_call.get('args')}")
                print(f"  ID: {tool_call.get('id')}")
        else:
            print("\nNo tool calls were made.")
            
        # Check additional kwargs for ASI-specific fields
        if hasattr(response, "additional_kwargs") and response.additional_kwargs:
            print("\nAdditional metadata:")
            for key, value in response.additional_kwargs.items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error in bind_tools test: {e}")
    
    # Test 2: with_structured_output
    print("\n[Test 2: with_structured_output]")
    try:
        # Create a model with structured output
        structured_model = asi_chat.with_structured_output(WeatherInfo)
        
        # Test the model with a query that should return structured data
        response = await structured_model.ainvoke([HumanMessage(content="What's the weather like in Paris?")])
        print(f"Structured response: {response}")
        
        # Test with include_raw=True
        structured_model_with_raw = asi_chat.with_structured_output(WeatherInfo, include_raw=True)
        response_with_raw = await structured_model_with_raw.ainvoke([HumanMessage(content="What's the weather like in London?")])
        print("\nStructured response with raw:")
        print(f"  Raw: {response_with_raw.get('raw')}")
        print(f"  Parsed: {response_with_raw.get('parsed')}")
        print(f"  Parsing error: {response_with_raw.get('parsing_error')}")
    except Exception as e:
        print(f"Error in with_structured_output test: {e}")
    
    # Test 3: ainvoke with different input formats
    print("\n[Test 3: ainvoke with different input formats]")
    try:
        # Test with HumanMessage in a list (preferred format)
        messages = [HumanMessage(content="What is the capital of France?")]
        messages_response = await asi_chat.ainvoke(messages)
        print(f"Messages input response: {messages_response.content}")
        
        # Test with single HumanMessage
        message_response = await asi_chat.ainvoke(HumanMessage(content="What is the capital of Italy?"))
        print(f"Message input response: {message_response.content}")
        
        # Test with string input (should be converted to HumanMessage)
        try:
            string_response = await asi_chat.ainvoke("What is the capital of Germany?")
            print(f"String input response: {string_response.content}")
        except Exception as e:
            print(f"String input error: {e}")
            print("Note: If this fails, the implementation may need to be updated to handle string inputs directly.")
    except Exception as e:
        print(f"Error in ainvoke test: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_advanced_features())
