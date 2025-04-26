"""
Parallel Tool Calling Comparison Test

This script compares parallel tool calling functionality between ASI and Groq implementations.
It tests how each provider handles multiple tool calls in a single response.
"""

import os
import asyncio
import json
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

async def test_parallel_tool_calling():
    """Test parallel tool calling functionality with both ASI and Groq."""
    print("=" * 80)
    print("PARALLEL TOOL CALLING COMPARISON TEST")
    print("=" * 80)
    
    # Initialize models
    asi_chat = ChatASI(model_name="asi1-mini")
    groq_chat = ChatGroq(model_name="llama3-8b-8192")
    
    # Define multiple tools
    tools = [
        {
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
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_restaurant",
                "description": "Find restaurants in a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "cuisine": {
                            "type": "string",
                            "description": "Type of cuisine, e.g. Italian, Chinese"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Test messages designed to trigger multiple tool calls
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="I'm planning a trip to New York. What's the weather like there, and can you recommend some Italian restaurants?")
    ]
    
    # Test ASI Parallel Tool Calling
    print("\n[Testing ASI Parallel Tool Calling]")
    try:
        asi_response = await asi_chat.agenerate(messages=[messages], tools=tools)
        print(f"ASI Response Content: {asi_response.generations[0][0].message.content}")
        
        # Check for tool calls
        tool_calls = asi_response.generations[0][0].message.tool_calls
        if tool_calls:
            print(f"\nASI made {len(tool_calls)} tool calls:")
            for i, tool_call in enumerate(tool_calls):
                print(f"  Tool Call {i+1}:")
                print(f"    Name: {tool_call.get('name')}")
                print(f"    Arguments: {json.dumps(tool_call.get('args'), indent=2)}")
                print(f"    ID: {tool_call.get('id')}")
        else:
            print("\nASI did not make any tool calls")
            
        # Check if ASI made parallel tool calls
        if len(tool_calls) > 1:
            print("\nASI successfully made parallel tool calls ✅")
        else:
            print("\nASI did not make parallel tool calls ❌")
    except Exception as e:
        print(f"ASI Error: {e}")
    
    # Test Groq Parallel Tool Calling
    print("\n[Testing Groq Parallel Tool Calling]")
    try:
        groq_response = await groq_chat.agenerate(messages=[messages], tools=tools)
        print(f"Groq Response Content: {groq_response.generations[0][0].message.content}")
        
        # Check for tool calls
        tool_calls = groq_response.generations[0][0].message.tool_calls
        if tool_calls:
            print(f"\nGroq made {len(tool_calls)} tool calls:")
            for i, tool_call in enumerate(tool_calls):
                print(f"  Tool Call {i+1}:")
                print(f"    Name: {tool_call.get('name')}")
                print(f"    Arguments: {json.dumps(tool_call.get('args'), indent=2)}")
                print(f"    ID: {tool_call.get('id')}")
        else:
            print("\nGroq did not make any tool calls")
            
        # Check if Groq made parallel tool calls
        if len(tool_calls) > 1:
            print("\nGroq successfully made parallel tool calls ✅")
        else:
            print("\nGroq did not make parallel tool calls ❌")
    except Exception as e:
        print(f"Groq Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_parallel_tool_calling())
