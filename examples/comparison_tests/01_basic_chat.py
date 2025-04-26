"""
Basic Chat Comparison Test

This script compares basic chat functionality between ASI and Groq implementations.
It tests simple message exchange with both providers to verify compatibility.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

async def test_basic_chat():
    """Test basic chat functionality with both ASI and Groq."""
    print("=" * 80)
    print("BASIC CHAT COMPARISON TEST")
    print("=" * 80)
    
    # Initialize models
    asi_chat = ChatASI(model_name="asi1-mini")
    groq_chat = ChatGroq(model_name="llama3-8b-8192")
    
    # Test messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    
    # Test ASI
    print("\n[Testing ASI]")
    try:
        asi_response = await asi_chat.agenerate(messages=[messages])
        print(f"ASI Response: {asi_response.generations[0][0].message.content}")
        print(f"ASI Token Usage: {asi_response.llm_output.get('token_usage', 'Not available')}")
    except Exception as e:
        print(f"ASI Error: {e}")
    
    # Test Groq
    print("\n[Testing Groq]")
    try:
        groq_response = await groq_chat.agenerate(messages=[messages])
        print(f"Groq Response: {groq_response.generations[0][0].message.content}")
        print(f"Groq Token Usage: {groq_response.llm_output.get('token_usage', 'Not available')}")
    except Exception as e:
        print(f"Groq Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_basic_chat())
