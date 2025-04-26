"""
Seed Parameter Comparison Test

This script compares seed parameter functionality between ASI and Groq implementations.
It tests how each provider handles the seed parameter for deterministic outputs.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_asi.chat_models import ChatASI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

async def test_seed_parameter():
    """Test seed parameter functionality with both ASI and Groq."""
    print("=" * 80)
    print("SEED PARAMETER COMPARISON TEST")
    print("=" * 80)
    
    # Initialize models
    seed_value = 42  # Fixed seed for deterministic outputs
    asi_chat = ChatASI(model_name="asi1-mini", temperature=0.0)
    groq_chat = ChatGroq(model_name="llama3-8b-8192", temperature=0.0)
    
    # Test messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Generate a random name for a pet dog.")
    ]
    
    # Test ASI with seed parameter
    print("\n[Testing ASI with Seed Parameter]")
    try:
        # First run with seed
        asi_response1 = await asi_chat.agenerate(messages=[messages], seed=seed_value)
        print(f"ASI Response (First Run): {asi_response1.generations[0][0].message.content}")
        
        # Second run with same seed
        asi_response2 = await asi_chat.agenerate(messages=[messages], seed=seed_value)
        print(f"ASI Response (Second Run): {asi_response2.generations[0][0].message.content}")
        
        # Check if responses are identical
        if asi_response1.generations[0][0].message.content == asi_response2.generations[0][0].message.content:
            print("\nASI produced identical responses with the same seed ✅")
        else:
            print("\nASI produced different responses despite using the same seed ❌")
    except Exception as e:
        print(f"ASI Error: {e}")
    
    # Test Groq with seed parameter
    print("\n[Testing Groq with Seed Parameter]")
    try:
        # First run with seed
        groq_response1 = await groq_chat.agenerate(messages=[messages], seed=seed_value)
        print(f"Groq Response (First Run): {groq_response1.generations[0][0].message.content}")
        
        # Second run with same seed
        groq_response2 = await groq_chat.agenerate(messages=[messages], seed=seed_value)
        print(f"Groq Response (Second Run): {groq_response2.generations[0][0].message.content}")
        
        # Check if responses are identical
        if groq_response1.generations[0][0].message.content == groq_response2.generations[0][0].message.content:
            print("\nGroq produced identical responses with the same seed ✅")
        else:
            print("\nGroq produced different responses despite using the same seed ❌")
    except Exception as e:
        print(f"Groq Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_seed_parameter())
