"""
Vision Capabilities Comparison Test

This script compares vision capabilities between ASI and Groq implementations.
It tests how each provider handles image inputs and generates descriptions.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

async def test_vision_capabilities():
    """Test vision capabilities with both ASI and Groq."""
    print("=" * 80)
    print("VISION CAPABILITIES COMPARISON TEST")
    print("=" * 80)
    
    # Initialize models
    # For ASI, use asi1-mini model which should support vision according to the user
    asi_chat = ChatASI(model_name="asi1-mini")
    # For Groq, use a model that might support vision
    groq_chat = ChatGroq(model_name="llama3-8b-8192")
    
    # Test image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    # Create multimodal message with image
    # Format 1: Using the standard multimodal format
    multimodal_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "What's in this image? Describe it in detail."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]
    )
    
    # Test ASI Vision
    print("\n[Testing ASI Vision]")
    try:
        asi_response = await asi_chat.agenerate(messages=[[multimodal_message]])
        print(f"ASI Response: {asi_response.generations[0][0].message.content}")
    except Exception as e:
        print(f"ASI Error: {e}")
    
    # Test Groq Vision
    print("\n[Testing Groq Vision]")
    try:
        groq_response = await groq_chat.agenerate(messages=[[multimodal_message]])
        print(f"Groq Response: {groq_response.generations[0][0].message.content}")
    except Exception as e:
        print(f"Groq Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_vision_capabilities())
