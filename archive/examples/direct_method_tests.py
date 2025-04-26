"""
Direct Method Tests for ChatASI

This script demonstrates how to use the core methods of the ChatASI class directly.
It tests both synchronous and asynchronous generation methods.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

async def test_direct_methods():
    """Test the direct use of _generate and _agenerate methods."""
    print("=" * 80)
    print("DIRECT METHOD TESTS")
    print("=" * 80)
    
    # Initialize the ChatASI model
    asi_chat = ChatASI(model_name="asi1-mini")
    
    # Test messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    
    # Test synchronous _generate method
    print("\n[Testing _generate method]")
    try:
        response = asi_chat._generate(messages=messages)
        print(f"Response: {response.generations[0].message.content}")
        print(f"Token Usage: {response.llm_output.get('token_usage', 'Not available')}")
        
        # Print additional metadata if available
        if hasattr(response.generations[0].message, "additional_kwargs"):
            additional_kwargs = response.generations[0].message.additional_kwargs
            if additional_kwargs:
                print("\nAdditional Metadata:")
                for key, value in additional_kwargs.items():
                    print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test asynchronous _agenerate method
    print("\n[Testing _agenerate method]")
    try:
        response = await asi_chat._agenerate(messages=messages)
        print(f"Response: {response.generations[0].message.content}")
        print(f"Token Usage: {response.llm_output.get('token_usage', 'Not available')}")
        
        # Print additional metadata if available
        if hasattr(response.generations[0].message, "additional_kwargs"):
            additional_kwargs = response.generations[0].message.additional_kwargs
            if additional_kwargs:
                print("\nAdditional Metadata:")
                for key, value in additional_kwargs.items():
                    print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with different parameters
    print("\n[Testing with custom parameters]")
    try:
        response = await asi_chat._agenerate(
            messages=messages,
            temperature=0.2,
            max_tokens=50
        )
        print(f"Response: {response.generations[0].message.content}")
        print(f"Token Usage: {response.llm_output.get('token_usage', 'Not available')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with stop sequences
    print("\n[Testing with stop sequences]")
    try:
        stop_sequences = [".", "!"]
        response = await asi_chat._agenerate(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Write a sentence about Paris")
            ],
            stop=stop_sequences
        )
        print(f"Response: {response.generations[0].message.content}")
        print(f"Token Usage: {response.llm_output.get('token_usage', 'Not available')}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_direct_methods())
