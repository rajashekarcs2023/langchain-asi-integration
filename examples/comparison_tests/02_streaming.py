"""
Streaming Comparison Test

This script compares streaming functionality between ASI and Groq implementations.
It tests the streaming of responses to verify compatibility and performance.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

async def test_streaming():
    """Test streaming functionality with both ASI and Groq."""
    print("=" * 80)
    print("STREAMING COMPARISON TEST")
    print("=" * 80)
    
    # Initialize models with streaming enabled
    asi_chat = ChatASI(model_name="asi1-mini", streaming=True)
    groq_chat = ChatGroq(model_name="llama3-8b-8192", streaming=True)
    
    # Test messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Write a short poem about artificial intelligence.")
    ]
    
    # Test ASI Streaming - End-user view
    print("\n[Testing ASI Streaming - End-user View]")
    try:
        print("ASI Response: ", end="", flush=True)
        async for chunk in asi_chat.astream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nASI Streaming Error: {e}")
        import traceback
        traceback.print_exc()
        
    # Test ASI Streaming - Debug view
    print("[Testing ASI Streaming - Debug View]")
    try:
        # Show detailed chunk information with character counts
        chunk_count = 0
        content_chunks = 0
        all_chunks = []
        async for chunk in asi_chat.astream(messages):
            chunk_count += 1
            if chunk.content:
                content_chunks += 1
                all_chunks.append(chunk.content)
                print(f"Chunk {chunk_count}: {len(chunk.content)} chars")
            else:
                print(f"Chunk {chunk_count}: [empty]")
        print(f"Total chunks: {chunk_count}, Content chunks: {content_chunks}")
        print(f"Average chunk size: {sum(len(c) for c in all_chunks)/max(1, len(all_chunks)):.1f} characters")
    except Exception as e:
        print(f"\nASI Streaming Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Groq Streaming - End-user view
    print("\n[Testing Groq Streaming - End-user View]")
    try:
        print("Groq Response: ", end="", flush=True)
        async for chunk in groq_chat.astream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nGroq Streaming Error: {e}")
        
    # Test Groq Streaming - Debug view
    print("[Testing Groq Streaming - Debug View]")
    try:
        # Show detailed chunk information with character counts
        chunk_count = 0
        content_chunks = 0
        all_chunks = []
        async for chunk in groq_chat.astream(messages):
            chunk_count += 1
            if chunk.content:
                content_chunks += 1
                all_chunks.append(chunk.content)
                print(f"Chunk {chunk_count}: {len(chunk.content)} chars")
            else:
                print(f"Chunk {chunk_count}: [empty]")
        print(f"Total chunks: {chunk_count}, Content chunks: {content_chunks}")
        print(f"Average chunk size: {sum(len(c) for c in all_chunks)/max(1, len(all_chunks)):.1f} characters")
    except Exception as e:
        print(f"\nGroq Streaming Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_streaming())
