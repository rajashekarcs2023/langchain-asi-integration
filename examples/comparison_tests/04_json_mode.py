"""
JSON Mode Comparison Test

This script compares JSON mode (structured output) functionality between ASI and Groq implementations.
It tests how each provider handles requests for structured JSON responses.
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

async def test_json_mode():
    """Test JSON mode functionality with both ASI and Groq."""
    print("=" * 80)
    print("JSON MODE COMPARISON TEST")
    print("=" * 80)
    
    # Initialize models
    asi_chat = ChatASI(model_name="asi1-mini")
    groq_chat = ChatGroq(model_name="llama3-8b-8192")
    
    # Test cases
    test_cases = [
        {
            "name": "Simple JSON object",
            "prompt": "Return a JSON object with your name and role."
        },
        {
            "name": "Complex nested structure",
            "prompt": "Return a JSON object representing a person with name, age, address (with street, city, country), and a list of 3 hobbies."
        }
    ]
    
    for test_case in test_cases:
        print(f"\n[Test Case: {test_case['name']}]")
        print(f"Prompt: {test_case['prompt']}")
        
        messages = [
            SystemMessage(content="You are a helpful assistant that responds in JSON format."),
            HumanMessage(content=test_case['prompt'])
        ]
        
        # Test ASI JSON Mode
        print("\n[Testing ASI JSON Mode]")
        try:
            asi_response = await asi_chat.agenerate(
                messages=[messages], 
                response_format={"type": "json_object"}
            )
            
            asi_content = asi_response.generations[0][0].message.content
            print(f"ASI Raw Response: {asi_content}")
            
            # Try to parse the JSON
            try:
                # For ASI, we might need to extract JSON from markdown code blocks
                if "```json" in asi_content:
                    json_pattern = r"```json\s*(.+?)\s*```"
                    import re
                    json_matches = re.findall(json_pattern, asi_content, re.DOTALL)
                    if json_matches:
                        asi_content = json_matches[0].strip()
                
                asi_parsed = json.loads(asi_content)
                print(f"ASI Parsed JSON: {json.dumps(asi_parsed, indent=2)}")
                print("ASI JSON parsing: SUCCESS ✅")
            except json.JSONDecodeError as e:
                print(f"ASI JSON parsing error: {e} ❌")
        except Exception as e:
            print(f"ASI Error: {e}")
        
        # Test Groq JSON Mode
        print("\n[Testing Groq JSON Mode]")
        try:
            groq_response = await groq_chat.agenerate(
                messages=[messages], 
                response_format={"type": "json_object"}
            )
            
            groq_content = groq_response.generations[0][0].message.content
            print(f"Groq Raw Response: {groq_content}")
            
            # Try to parse the JSON
            try:
                groq_parsed = json.loads(groq_content)
                print(f"Groq Parsed JSON: {json.dumps(groq_parsed, indent=2)}")
                print("Groq JSON parsing: SUCCESS ✅")
            except json.JSONDecodeError as e:
                print(f"Groq JSON parsing error: {e} ❌")
        except Exception as e:
            print(f"Groq Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_json_mode())
