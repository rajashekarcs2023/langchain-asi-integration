"""
Basic Message Test for ChatASI

This script tests the basic message handling functionality of the ChatASI class.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

async def test_basic_message_handling():
    """Test basic message handling in the ChatASI class."""
    print("=" * 80)
    print("BASIC MESSAGE HANDLING TEST")
    print("=" * 80)
    
    # Initialize the ChatASI model
    asi_chat = ChatASI(model_name="asi1-mini", verbose=True)
    
    # Test 1: Basic message list with system and human messages
    print("\n[Test 1: Basic message list]")
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?")
        ]
        
        # Print the message objects for debugging
        print("Message objects:")
        for i, msg in enumerate(messages):
            print(f"  Message {i}: {type(msg).__name__}, content: {msg.content}")
        
        # Convert messages to dicts using the internal method
        message_dicts = asi_chat._create_message_dicts(messages)
        
        print("\nConverted message dicts:")
        for i, msg_dict in enumerate(message_dicts):
            print(f"  Dict {i}: role: {msg_dict.get('role')}, content: {msg_dict.get('content')}")
        
        # Test the agenerate method
        response = await asi_chat.agenerate(messages=[messages])
        print(f"\nResponse: {response.generations[0].message.content}")
    except Exception as e:
        print(f"Error in basic message list test: {e}")
    
    # Test 2: Single human message
    print("\n[Test 2: Single human message]")
    try:
        message = HumanMessage(content="What is the capital of Italy?")
        
        # Print the message object for debugging
        print(f"Message object: {type(message).__name__}, content: {message.content}")
        
        # Convert message to dict using the internal method
        message_dict = asi_chat._convert_message_to_dict(message)
        
        print(f"\nConverted message dict: role: {message_dict.get('role')}, content: {message_dict.get('content')}")
        
        # Test the agenerate method
        response = await asi_chat.agenerate(messages=[[message]])
        print(f"\nResponse: {response.generations[0].message.content}")
    except Exception as e:
        print(f"Error in single human message test: {e}")
    
    # Test 3: Message tuple format
    print("\n[Test 3: Message tuple format]")
    try:
        message_tuple = ("content", "What is the capital of Germany?")
        
        # Print the message tuple for debugging
        print(f"Message tuple: {message_tuple}")
        
        # Convert message tuple to dict using the internal method
        try:
            message_dict = asi_chat._convert_message_to_dict(message_tuple)
            print(f"\nConverted message dict: role: {message_dict.get('role')}, content: {message_dict.get('content')}")
        except Exception as e:
            print(f"Error converting message tuple to dict: {e}")
        
        # Try to use the message tuple with agenerate
        try:
            response = await asi_chat.agenerate(messages=[[message_tuple]])
            print(f"\nResponse: {response.generations[0].message.content}")
        except Exception as e:
            print(f"Error using message tuple with agenerate: {e}")
    except Exception as e:
        print(f"Error in message tuple format test: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_basic_message_handling())
