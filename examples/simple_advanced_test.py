"""
Simple Advanced Features Test for ChatASI

This script tests basic functionality of advanced features in the ChatASI class
with simpler, more direct approaches.
"""

import os
import asyncio
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Define a simple Pydantic model for structured output
class SimpleResponse(BaseModel):
    """A simple response with a message."""
    message: str = Field(description="A simple message response")

# Define a simple tool
class SimpleCalculator(BaseModel):
    """A simple calculator tool."""
    a: int = Field(description="First number")
    b: int = Field(description="Second number")
    operation: str = Field(description="Operation to perform (add, subtract, multiply, divide)")

async def test_simple_advanced_features():
    """Test basic functionality of advanced features in the ChatASI class."""
    print("=" * 80)
    print("SIMPLE ADVANCED FEATURES TEST")
    print("=" * 80)
    
    # Initialize the ChatASI model
    asi_chat = ChatASI(model_name="asi1-mini", verbose=True)
    
    # Test basic invoke functionality
    print("\n[Test 1: Basic invoke]")
    try:
        # Test with a list of messages (standard format)
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?")
        ]
        response = await asi_chat.ainvoke(messages)
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error in basic invoke test: {e}")
    
    # Test basic structured output
    print("\n[Test 2: Basic structured output]")
    try:
        # Create a model with structured output
        structured_model = asi_chat.with_structured_output(SimpleResponse)
        
        # Test with the standard message format
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Say hello to the world.")
        ]
        response = await structured_model.ainvoke(messages)
        print(f"Structured response: {response}")
    except Exception as e:
        print(f"Error in basic structured output test: {e}")
    
    # Test basic tool binding
    print("\n[Test 3: Basic tool binding]")
    try:
        # Bind a simple tool
        model_with_tool = asi_chat.bind_tools([SimpleCalculator])
        
        # Test with the standard message format
        messages = [
            SystemMessage(content="You are a helpful assistant that can perform calculations."),
            HumanMessage(content="Calculate 5 + 3")
        ]
        response = await model_with_tool.ainvoke(messages)
        print(f"Response: {response.content}")
        
        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            print("\nTool calls:")
            for tool_call in response.tool_calls:
                print(f"  Tool: {tool_call.get('name')}")
                print(f"  Arguments: {tool_call.get('args')}")
        else:
            print("\nNo tool calls were made.")
    except Exception as e:
        print(f"Error in basic tool binding test: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_simple_advanced_features())
