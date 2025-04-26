"""
Retry Mechanism Tests for ChatASI

This script tests the retry mechanisms in the ChatASI class.
It specifically tests the completion_with_retry and acompletion_with_retry methods.
"""

import os
import asyncio
import time
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

async def test_retry_mechanisms():
    """Test the retry mechanisms in the ChatASI class."""
    print("=" * 80)
    print("RETRY MECHANISM TESTS")
    print("=" * 80)
    
    # Initialize the ChatASI model with verbose logging
    asi_chat = ChatASI(model_name="asi1-mini", verbose=True, max_retries=2)
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # Test synchronous completion_with_retry
    print("\n[Testing completion_with_retry]")
    try:
        start_time = time.time()
        response = asi_chat.completion_with_retry(messages=messages)
        end_time = time.time()
        
        print(f"Response received in {end_time - start_time:.2f} seconds")
        print(f"Model: {response.get('model', 'Not specified')}")
        
        # Check for choices in the response
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            print(f"Response: {choice.get('message', {}).get('content', 'No content')}")
            
            # Check for additional fields
            if "finish_reason" in choice:
                print(f"Finish reason: {choice['finish_reason']}")
                
        # Check for usage information
        if "usage" in response:
            usage = response["usage"]
            print(f"Token usage: {usage}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test asynchronous acompletion_with_retry
    print("\n[Testing acompletion_with_retry]")
    try:
        start_time = time.time()
        response = await asi_chat.acompletion_with_retry(messages=messages)
        end_time = time.time()
        
        print(f"Response received in {end_time - start_time:.2f} seconds")
        print(f"Model: {response.get('model', 'Not specified')}")
        
        # Check for choices in the response
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            print(f"Response: {choice.get('message', {}).get('content', 'No content')}")
            
            # Check for additional fields
            if "finish_reason" in choice:
                print(f"Finish reason: {choice['finish_reason']}")
                
        # Check for usage information
        if "usage" in response:
            usage = response["usage"]
            print(f"Token usage: {usage}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with invalid model to trigger retry
    print("\n[Testing retry with invalid model]")
    invalid_model_chat = ChatASI(model_name="invalid-model", verbose=True, max_retries=1)
    try:
        await invalid_model_chat.acompletion_with_retry(messages=messages)
    except Exception as e:
        print(f"Expected error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_retry_mechanisms())
