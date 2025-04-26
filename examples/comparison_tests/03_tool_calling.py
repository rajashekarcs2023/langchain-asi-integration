"""
Tool Calling Comparison Test

This script compares tool calling functionality between ASI and Groq implementations.
It tests how each provider handles tool definitions and makes tool calls.
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

async def test_tool_calling():
    """Test tool calling functionality with both ASI and Groq."""
    print("=" * 80)
    print("TOOL CALLING COMPARISON TEST")
    print("=" * 80)
    
    # Initialize models
    asi_chat = ChatASI(model_name="asi1-mini")
    groq_chat = ChatGroq(model_name="llama3-8b-8192")
    
    # Define tools
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
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Test messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What's the weather like in New York?")
    ]
    
    # Test ASI Tool Calling
    print("\n[Testing ASI Tool Calling]")
    try:
        asi_response = await asi_chat.agenerate(messages=[messages], tools=tools)
        print(f"ASI Response Content: {asi_response.generations[0][0].message.content}")
        
        # Check for tool calls
        tool_calls = asi_response.generations[0][0].message.tool_calls
        if tool_calls:
            print("\nASI Tool Calls:")
            for tool_call in tool_calls:
                print(f"  Tool: {tool_call.get('name')}")
                print(f"  Arguments: {json.dumps(tool_call.get('args'), indent=2)}")
                print(f"  ID: {tool_call.get('id')}")
        else:
            print("\nASI did not make any tool calls")
            
        # Check for additional kwargs that might contain ASI-specific information
        if asi_response.generations[0][0].message.additional_kwargs:
            print("\nASI Additional Info:")
            for key, value in asi_response.generations[0][0].message.additional_kwargs.items():
                if key in ['thought', 'tool_thought']:
                    print(f"  {key}: {value}")
    except Exception as e:
        print(f"ASI Error: {e}")
    
    # Test Groq Tool Calling
    print("\n[Testing Groq Tool Calling]")
    try:
        groq_response = await groq_chat.agenerate(messages=[messages], tools=tools)
        print(f"Groq Response Content: {groq_response.generations[0][0].message.content}")
        
        # Check for tool calls
        tool_calls = groq_response.generations[0][0].message.tool_calls
        if tool_calls:
            print("\nGroq Tool Calls:")
            for tool_call in tool_calls:
                print(f"  Tool: {tool_call.get('name')}")
                print(f"  Arguments: {json.dumps(tool_call.get('args'), indent=2)}")
                print(f"  ID: {tool_call.get('id')}")
        else:
            print("\nGroq did not make any tool calls")
    except Exception as e:
        print(f"Groq Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_tool_calling())
