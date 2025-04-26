"""
Synchronous Streaming Comparison Test

This script compares synchronous streaming functionality between ASI and Groq implementations.
It tests the streaming of responses to verify compatibility and performance.
"""

import os
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

def test_sync_streaming():
    """Test synchronous streaming functionality with both ASI and Groq."""
    print("=" * 80)
    print("SYNCHRONOUS STREAMING COMPARISON TEST")
    print("=" * 80)
    
    # Initialize models with streaming enabled
    asi_chat = ChatASI(model_name="asi1-mini", streaming=True)
    groq_chat = ChatGroq(model_name="llama3-8b-8192", streaming=True)
    
    # Test messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Write a short poem about artificial intelligence.")
    ]
    
    # Test ASI Synchronous Streaming
    print("\n[Testing ASI Synchronous Streaming]")
    try:
        print("ASI Response: ", end="", flush=True)
        for chunk in asi_chat.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nASI Streaming Error: {e}")
    
    # Test Groq Synchronous Streaming
    print("\n[Testing Groq Synchronous Streaming]")
    try:
        print("Groq Response: ", end="", flush=True)
        for chunk in groq_chat.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nGroq Streaming Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_sync_streaming()
