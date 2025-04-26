"""
Debug Bind Tools Test

This script debugs the bind_tools method in the ChatASI class to identify
where the issue is occurring.
"""

import os
import asyncio
import inspect
import traceback
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Union, Sequence, Type, Literal
from pydantic import BaseModel, Field
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.language_models import LanguageModelInput

# Load environment variables
load_dotenv()

# Define a simple tool
class GetWeather(BaseModel):
    """Get the current weather in a given location."""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

def debug_print(prefix, obj):
    """Print debug information about an object."""
    print(f"{prefix}: {type(obj)}")
    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            print(f"  {key}: {type(value)}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            print(f"  {key}: {type(value)}")
    elif isinstance(obj, list):
        print(f"  List with {len(obj)} items")
        for i, item in enumerate(obj[:3]):  # Show first 3 items
            print(f"    Item {i}: {type(item)}")
    else:
        print(f"  Value: {obj}")

async def debug_bind_tools():
    """Debug the bind_tools method in the ChatASI class."""
    print("=" * 80)
    print("DEBUG BIND TOOLS")
    print("=" * 80)
    
    # Initialize the ChatASI model
    asi_chat = ChatASI(model_name="asi1-mini", verbose=True)
    
    # Print the class hierarchy
    print("\nClass hierarchy:")
    for cls in asi_chat.__class__.__mro__:
        print(f"  {cls.__name__}")
    
    # Check if bind_tools is available
    print("\nChecking for bind_tools method:")
    if hasattr(asi_chat, "bind_tools"):
        print(f"  bind_tools method exists: {asi_chat.bind_tools}")
        print(f"  Source: {inspect.getsource(asi_chat.bind_tools)}")
    else:
        print("  bind_tools method does not exist")
        # Check if it's available in parent classes
        for cls in asi_chat.__class__.__mro__[1:]:  # Skip the class itself
            if hasattr(cls, "bind_tools"):
                print(f"  bind_tools method exists in parent class {cls.__name__}")
                print(f"  Source: {inspect.getsource(cls.bind_tools)}")
                break
    
    # Check if bind method is available
    print("\nChecking for bind method:")
    if hasattr(asi_chat, "bind"):
        print(f"  bind method exists: {asi_chat.bind}")
        try:
            print(f"  Source: {inspect.getsource(asi_chat.bind)}")
        except (TypeError, OSError):
            print("  Could not get source code for bind method")
    else:
        print("  bind method does not exist")
    
    # Try to use bind_tools
    print("\nTrying to use bind_tools:")
    try:
        # Bind tools to the model
        model_with_tools = asi_chat.bind_tools([GetWeather])
        
        # Print information about the model with tools
        print("  Successfully created model with tools")
        debug_print("  Model with tools", model_with_tools)
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant that can provide weather information."),
            HumanMessage(content="What's the weather like in San Francisco?")
        ]
        
        # Print information about the messages
        print("\nMessages:")
        for i, msg in enumerate(messages):
            debug_print(f"  Message {i}", msg)
        
        # Try to invoke the model
        print("\nTrying to invoke the model:")
        try:
            # Use the model with tools
            response = await model_with_tools.ainvoke(messages)
            print("  Successfully invoked model")
            debug_print("  Response", response)
        except Exception as e:
            print(f"  Error invoking model: {e}")
            traceback.print_exc()
            
            # Try to identify where the error is occurring
            print("\nDebugging invoke error:")
            try:
                # Convert messages to the format expected by the API
                print("  Converting messages to API format...")
                message_dicts = asi_chat._create_message_dicts(messages)
                print(f"  Successfully converted messages: {message_dicts}")
                
                # Try to make a direct API call
                print("  Making direct API call...")
                tools = [convert_to_openai_tool(GetWeather)]
                print(f"  Tools: {tools}")
                
                # Format tools for ASI API
                asi_tools = []
                for tool in tools:
                    if "function" in tool:
                        function_def = tool["function"]
                        asi_tools.append({
                            "type": "function",
                            "function": function_def
                        })
                print(f"  ASI tools: {asi_tools}")
                
                # Make the API call
                response = await asi_chat.acompletion_with_retry(
                    messages=message_dicts,
                    tools=asi_tools,
                    tool_choice="auto"
                )
                print("  Successfully made direct API call")
                print(f"  Response: {response}")
            except Exception as e2:
                print(f"  Error in debugging: {e2}")
                traceback.print_exc()
    except Exception as e:
        print(f"  Error using bind_tools: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(debug_bind_tools())
