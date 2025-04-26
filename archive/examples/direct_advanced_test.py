"""
Direct Advanced Test for ChatASI

This script directly tests the advanced features of the ChatASI class
without using the higher-level LangChain abstractions.
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

async def test_direct_advanced_features():
    """Test advanced features of the ChatASI class directly."""
    print("=" * 80)
    print("DIRECT ADVANCED FEATURES TEST")
    print("=" * 80)
    
    # Initialize the ChatASI model
    asi_chat = ChatASI(model_name="asi1-mini", verbose=True)
    
    # Test 1: Direct tool calling
    print("\n[Test 1: Direct tool calling]")
    try:
        # Define a tool schema directly
        calculator_tool = {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "A simple calculator",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]}
                    },
                    "required": ["a", "b", "operation"]
                }
            }
        }
        
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can perform calculations."},
            {"role": "user", "content": "Calculate 5 + 3"}
        ]
        
        # Make a direct API call with tools
        response = await asi_chat.acompletion_with_retry(
            messages=messages,
            tools=[calculator_tool],
            tool_choice="auto"
        )
        
        print(f"Response: {response}")
        
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
        print(f"Error in direct tool calling test: {e}")
    
    # Test 2: Direct JSON mode
    print("\n[Test 2: Direct JSON mode]")
    try:
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides structured responses."},
            {"role": "user", "content": "Provide a greeting message."}
        ]
        
        # Make a direct API call with JSON mode
        response = await asi_chat.acompletion_with_retry(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        print(f"Response: {response}")
        
        # Check for JSON content
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content", "")
            print(f"\nJSON content: {content}")
    except Exception as e:
        print(f"Error in direct JSON mode test: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_direct_advanced_features())
