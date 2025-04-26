"""Test script to compare raw API responses from Groq and ASI for tool calling."""

import os
import json
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ASI_API_KEY = os.getenv("ASI_API_KEY")

# API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
ASI_API_URL = "https://api.asi1.ai/v1/chat/completions"

# Define a simple tool schema
route_tool = {
    "type": "function",
    "function": {
        "name": "route",
        "description": "Select the next role based on query analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "next": {
                    "type": "string",
                    "enum": ["FINISH", "Search", "SECAnalyst"],
                    "description": "The next agent to act"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation for why this agent should act next"
                },
                "information_needed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific information needed from this agent"
                }
            },
            "required": ["next", "reasoning", "information_needed"],
        }
    }
}

# Test messages
messages = [
    {"role": "system", "content": "You are a helpful assistant that routes queries to the appropriate agent."},
    {"role": "user", "content": "What are Apple's key financial risks and how have they changed over the past year?"}
]

def test_groq_api():
    """Test Groq's API for tool calling."""
    print("\n=== Testing Groq API ===")
    
    # Prepare request data
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "tools": [route_tool],
        "tool_choice": {"type": "function", "function": {"name": "route"}},
        "temperature": 0
    }
    
    # Set headers
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Make the API request
    print("Sending request to Groq API...")
    try:
        with httpx.Client(timeout=30.0) as client:  # Use a client with timeout
            response = client.post(GROQ_API_URL, json=data, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nGroq API Response:")
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            print(f"Response body (pretty):")
            print(json.dumps(result, indent=2))
            
            # Extract and print the assistant's message
            assistant_message = result["choices"][0]["message"]
            print("\nAssistant Message:")
            print(json.dumps(assistant_message, indent=2))
            
            # Check for tool calls
            if "tool_calls" in assistant_message:
                print("\nTool Calls:")
                print(json.dumps(assistant_message["tool_calls"], indent=2))
            else:
                print("\nNo tool_calls found in the response.")
                print("Content:", assistant_message.get("content", "No content"))
                print("Finish reason:", result["choices"][0].get("finish_reason", "Not specified"))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error making request to Groq API: {str(e)}")

def test_asi_api():
    """Test ASI's API for tool calling."""
    print("\n=== Testing ASI API ===")
    
    # Prepare request data
    data = {
        "model": "asi1-mini",
        "messages": messages,
        "tools": [route_tool],
        "tool_choice": {"type": "function", "function": {"name": "route"}},
        "temperature": 0
    }
    
    # Set headers
    headers = {
        "Authorization": f"Bearer {ASI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Make the API request with increased timeout
    print("Sending request to ASI API...")
    try:
        with httpx.Client(timeout=60.0) as client:  # Increase timeout to 60 seconds
            response = client.post(ASI_API_URL, json=data, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nASI API Response:")
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            print(f"Response body (pretty):")
            print(json.dumps(result, indent=2))
            
            # Extract and print the assistant's message
            assistant_message = result["choices"][0]["message"]
            print("\nAssistant Message:")
            print(json.dumps(assistant_message, indent=2))
            
            # Check for tool calls
            if "tool_calls" in assistant_message:
                print("\nTool Calls:")
                print(json.dumps(assistant_message["tool_calls"], indent=2))
            else:
                print("\nNo tool_calls found in the response.")
                print("Content:", assistant_message.get("content", "No content"))
                print("Finish reason:", result["choices"][0].get("finish_reason", "Not specified"))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error making request to ASI API: {str(e)}")

def test_asi_api_json_mode():
    """Test ASI's API with JSON mode."""
    print("\n=== Testing ASI API with JSON Mode ===")
    
    # Prepare request data
    data = {
        "model": "asi1-mini",
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0
    }
    
    # Set headers
    headers = {
        "Authorization": f"Bearer {ASI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Make the API request with increased timeout
    print("Sending request to ASI API with JSON mode...")
    try:
        with httpx.Client(timeout=60.0) as client:  # Increase timeout to 60 seconds
            response = client.post(ASI_API_URL, json=data, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nASI API Response (JSON Mode):")
            print(f"Status code: {response.status_code}")
            print(f"Response body (pretty):")
            print(json.dumps(result, indent=2))
            
            # Extract and print the assistant's message
            assistant_message = result["choices"][0]["message"]
            print("\nAssistant Message:")
            print(json.dumps(assistant_message, indent=2))
            
            # Check if the content is valid JSON
            content = assistant_message.get("content", "")
            print("\nContent:")
            print(content)
            
            try:
                if content:
                    json_content = json.loads(content)
                    print("\nParsed JSON Content:")
                    print(json.dumps(json_content, indent=2))
            except json.JSONDecodeError as e:
                print(f"\nContent is not valid JSON: {str(e)}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error making request to ASI API: {str(e)}")

if __name__ == "__main__":
    # Test both APIs
    test_groq_api()
    test_asi_api()
    test_asi_api_json_mode()
