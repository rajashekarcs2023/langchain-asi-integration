"""
Explore ASI API capabilities by testing different methods and features.
This script will help us understand what the ASI API supports.
"""

import os
import json
import asyncio
import httpx
from dotenv import load_dotenv
from pprint import pprint

# Load environment variables
load_dotenv()

# Get API key from environment
ASI_API_KEY = os.getenv("ASI_API_KEY")
ASI_API_BASE = "https://api.asi1.ai/v1"  # For asi1-mini model

async def test_basic_chat():
    """Test basic chat functionality with the ASI API."""
    print("\n" + "=" * 80)
    print(" Testing Basic Chat ")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Prepare the request
        url = f"{ASI_API_BASE}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ASI_API_KEY}",
            "Accept": "application/json"
        }
        data = {
            "model": "asi1-mini",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        # Make the request
        response = await client.post(url, headers=headers, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            print("Response status: Success")
            print("\nResponse structure:")
            pprint(response_data)
            
            # Extract and print the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print("\nResponse content:")
                print(content)
            
            # Print token usage information
            if "usage" in response_data:
                print("\nToken usage:")
                pprint(response_data["usage"])
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

async def test_system_messages():
    """Test system message handling with the ASI API."""
    print("\n" + "=" * 80)
    print(" Testing System Messages ")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Prepare the request
        url = f"{ASI_API_BASE}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ASI_API_KEY}",
            "Accept": "application/json"
        }
        data = {
            "model": "asi1-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
                {"role": "user", "content": "Tell me about the weather today."}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        # Make the request
        response = await client.post(url, headers=headers, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            print("Response status: Success")
            print("\nResponse structure:")
            pprint(response_data)
            
            # Extract and print the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print("\nResponse content:")
                print(content)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

async def test_streaming():
    """Test streaming functionality with the ASI API."""
    print("\n" + "=" * 80)
    print(" Testing Streaming ")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Prepare the request
        url = f"{ASI_API_BASE}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ASI_API_KEY}",
            "Accept": "application/json"
        }
        data = {
            "model": "asi1-mini",
            "messages": [
                {"role": "user", "content": "Write a short poem about artificial intelligence."}
            ],
            "temperature": 0.7,
            "max_tokens": 150,
            "stream": True
        }
        
        # Make the request
        async with client.stream("POST", url, headers=headers, json=data) as response:
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                print("\nStreaming response:")
                full_content = ""
                async for chunk in response.aiter_lines():
                    if chunk.strip():
                        # Remove the "data: " prefix if it exists
                        if chunk.startswith("data: "):
                            chunk = chunk[6:]
                        
                        # Skip "data: [DONE]" message
                        if chunk == "[DONE]":
                            continue
                        
                        try:
                            chunk_data = json.loads(chunk)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    full_content += content
                                    print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            print(f"Error parsing chunk: {chunk}")
                
                print("\n\nFull content:")
                print(full_content)
            else:
                print(f"Error: {response.status_code}")
                print(response.text)

async def test_tool_calling():
    """Test tool calling functionality with the ASI API."""
    print("\n" + "=" * 80)
    print(" Testing Tool Calling ")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Prepare the request
        url = f"{ASI_API_BASE}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ASI_API_KEY}",
            "Accept": "application/json"
        }
        data = {
            "model": "asi1-mini",
            "messages": [
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            "temperature": 0.7,
            "max_tokens": 150,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "The unit of temperature",
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ]
        }
        
        # Make the request
        response = await client.post(url, headers=headers, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            print("Response status: Success")
            print("\nResponse structure:")
            pprint(response_data)
            
            # Extract and print the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                message = response_data["choices"][0]["message"]
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])
                
                print("\nResponse content:")
                print(content)
                
                if tool_calls:
                    print("\nTool calls:")
                    for tool_call in tool_calls:
                        print(f"Tool ID: {tool_call.get('id')}")
                        print(f"Tool Type: {tool_call.get('type')}")
                        function = tool_call.get("function", {})
                        print(f"Function Name: {function.get('name')}")
                        print(f"Function Arguments: {function.get('arguments')}")
                        
                        # Try to parse the arguments as JSON
                        try:
                            args = json.loads(function.get("arguments", "{}"))
                            print("Parsed Arguments:")
                            pprint(args)
                        except json.JSONDecodeError:
                            print("Could not parse arguments as JSON")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

async def test_json_mode():
    """Test JSON mode functionality with the ASI API."""
    print("\n" + "=" * 80)
    print(" Testing JSON Mode ")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Prepare the request
        url = f"{ASI_API_BASE}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ASI_API_KEY}",
            "Accept": "application/json"
        }
        data = {
            "model": "asi1-mini",
            "messages": [
                {"role": "user", "content": "Generate a JSON object with the following fields: name, age, and city. Make up the values."}
            ],
            "temperature": 0.7,
            "max_tokens": 150,
            "response_format": {"type": "json_object"}
        }
        
        # Make the request
        response = await client.post(url, headers=headers, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            print("Response status: Success")
            print("\nResponse structure:")
            pprint(response_data)
            
            # Extract and print the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print("\nResponse content:")
                print(content)
                
                # Try to parse the content as JSON
                try:
                    json_content = json.loads(content)
                    print("\nParsed JSON:")
                    pprint(json_content)
                except json.JSONDecodeError:
                    print("Could not parse content as JSON")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

async def test_seed_parameter():
    """Test seed parameter functionality with the ASI API."""
    print("\n" + "=" * 80)
    print(" Testing Seed Parameter ")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Prepare the request
        url = f"{ASI_API_BASE}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ASI_API_KEY}",
            "Accept": "application/json"
        }
        
        # First request with seed
        seed_value = 42
        data = {
            "model": "asi1-mini",
            "messages": [
                {"role": "user", "content": "Generate a random number between 1 and 100."}
            ],
            "temperature": 0.7,
            "max_tokens": 50,
            "seed": seed_value
        }
        
        # Make the first request
        response1 = await client.post(url, headers=headers, json=data)
        
        # Make the second request with the same seed
        response2 = await client.post(url, headers=headers, json=data)
        
        # Check if the requests were successful
        if response1.status_code == 200 and response2.status_code == 200:
            response_data1 = response1.json()
            response_data2 = response2.json()
            
            print("First response:")
            content1 = response_data1["choices"][0]["message"]["content"]
            print(content1)
            
            print("\nSecond response (with same seed):")
            content2 = response_data2["choices"][0]["message"]["content"]
            print(content2)
            
            print("\nAre responses identical?", content1 == content2)
        else:
            print(f"Error in first request: {response1.status_code}")
            print(f"Error in second request: {response2.status_code}")

async def test_parallel_function_calling():
    """Test parallel function calling with the ASI API."""
    print("\n" + "=" * 80)
    print(" Testing Parallel Function Calling ")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Prepare the request
        url = f"{ASI_API_BASE}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ASI_API_KEY}",
            "Accept": "application/json"
        }
        data = {
            "model": "asi1-mini",
            "messages": [
                {"role": "user", "content": "What's the weather like in San Francisco and New York?"}
            ],
            "temperature": 0.7,
            "max_tokens": 150,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "The unit of temperature",
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ]
        }
        
        # Make the request
        response = await client.post(url, headers=headers, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            print("Response status: Success")
            print("\nResponse structure:")
            pprint(response_data)
            
            # Extract and print the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                message = response_data["choices"][0]["message"]
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])
                
                print("\nResponse content:")
                print(content)
                
                if tool_calls:
                    print("\nTool calls:")
                    for tool_call in tool_calls:
                        print(f"Tool ID: {tool_call.get('id')}")
                        print(f"Tool Type: {tool_call.get('type')}")
                        function = tool_call.get("function", {})
                        print(f"Function Name: {function.get('name')}")
                        print(f"Function Arguments: {function.get('arguments')}")
                        
                        # Try to parse the arguments as JSON
                        try:
                            args = json.loads(function.get("arguments", "{}"))
                            print("Parsed Arguments:")
                            pprint(args)
                        except json.JSONDecodeError:
                            print("Could not parse arguments as JSON")
                    
                    # Check if multiple tool calls were made
                    print(f"\nNumber of tool calls: {len(tool_calls)}")
                    if len(tool_calls) > 1:
                        print("Parallel function calling is supported!")
                    else:
                        print("Only a single function call was made.")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

async def test_vision_capabilities():
    """Test vision capabilities with the ASI API."""
    print("\n" + "=" * 80)
    print(" Testing Vision Capabilities ")
    print("=" * 80)
    
    # Image URL to test with
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Prepare the request
        url = f"{ASI_API_BASE}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ASI_API_KEY}",
            "Accept": "application/json"
        }
        data = {
            "model": "asi1-mini",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        # Make the request
        response = await client.post(url, headers=headers, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            print("Response status: Success")
            print("\nResponse structure:")
            pprint(response_data)
            
            # Extract and print the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print("\nResponse content:")
                print(content)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            print("\nVision capabilities may not be supported.")

async def main():
    """Run all the tests."""
    print("=" * 80)
    print(" Exploring ASI API Capabilities ")
    print("=" * 80)
    
    # Test basic chat
    await test_basic_chat()
    
    # Test system messages
    await test_system_messages()
    
    # Test streaming
    await test_streaming()
    
    # Test tool calling
    await test_tool_calling()
    
    # Test JSON mode
    await test_json_mode()
    
    # Test seed parameter
    await test_seed_parameter()
    
    # Test parallel function calling
    await test_parallel_function_calling()
    
    # Test vision capabilities
    await test_vision_capabilities()
    
    print("\n" + "=" * 80)
    print(" Testing Complete ")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
