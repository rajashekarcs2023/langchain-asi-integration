import os
import json
import httpx
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for API keys
ASI_API_KEY = os.environ.get("ASI_API_KEY")
if not ASI_API_KEY:
    print("Error: ASI_API_KEY environment variable is not set")
    print("Please set it in your .env file or export it in your terminal")
    sys.exit(1)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set")
    print("Please set it in your .env file or export it in your terminal")
    sys.exit(1)

# API base URLs and models
ASI_API_BASE = os.environ.get("ASI_API_BASE", "https://api.asi1.ai")
ASI_MODEL = os.environ.get("ASI_MODEL", "asi1-mini")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

print(f"Using ASI API base: {ASI_API_BASE}")
print(f"Using ASI model: {ASI_MODEL}")
print(f"Using OpenAI model: {OPENAI_MODEL}")

# Define a simple tool for weather
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use"
                }
            },
            "required": ["location"]
        }
    }
}

# Mock ASI API response for regular completion
mock_asi_regular_response = {
    "id": "asi-123",
    "model": "asi1-mini",
    "thought": ["I should greet the user politely and ask how they are doing."],
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello! I'm doing well, thank you for asking. How are you today?"
            }
        }
    ]
}

# Mock OpenAI API response for regular completion
mock_openai_regular_response = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm doing well, thank you for asking. How are you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 15,
        "total_tokens": 28
    }
}

# Mock ASI API response for tool calling
mock_asi_tool_response = {
    "id": "asi-456",
    "model": "asi1-mini",
    "thought": ["The user is asking about weather in San Francisco. I should use the get_weather function to retrieve this information."],
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
                        }
                    }
                ]
            }
        }
    ]
}

# Mock OpenAI API response for tool calling
mock_openai_tool_response = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 25,
        "total_tokens": 45
    }
}

# Mock ASI API response for structured output
mock_asi_structured_response = {
    "id": "asi-789",
    "model": "asi1-mini",
    "thought": ["I need to create a JSON object with name, age, and occupation for John Doe."],
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "{\"name\": \"John Doe\", \"age\": 30, \"occupation\": \"Software Engineer\"}"
            }
        }
    ]
}

# Mock OpenAI API response for structured output
mock_openai_structured_response = {
    "id": "chatcmpl-789",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"name\": \"John Doe\", \"age\": 30, \"occupation\": \"Software Engineer\"}"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 30,
        "completion_tokens": 20,
        "total_tokens": 50
    }
}

# Function to make a direct API call to ASI
def call_asi_api(messages: List[Dict[str, Any]], tools=None, response_format=None) -> Dict[str, Any]:
    """Make a direct call to the ASI API and return the raw response."""
    url = f"{ASI_API_BASE}/v1/chat/completions"  
    headers = {
        "Authorization": f"Bearer {ASI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    data = {
        "model": ASI_MODEL,  
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    if tools:
        data["tools"] = tools
    
    if response_format:
        data["response_format"] = response_format
    
    print(f"\nASI API Request to {url}:\n{json.dumps(data, indent=2)}")
    
    try:
        # For demonstration, use mock responses instead of actual API calls
        if tools and any(tool.get("function", {}).get("name") == "get_weather" for tool in tools):
            return mock_asi_tool_response
        elif response_format and response_format.get("type") == "json_object":
            return mock_asi_structured_response
        elif len(messages) > 1 and any(msg.get("role") == "function" for msg in messages):
            # This is a function response scenario
            return mock_asi_regular_response
        else:
            return mock_asi_regular_response
    except Exception as e:
        print(f"ASI API Error: {str(e)}")
        return {}

# Function to make a direct API call to OpenAI
def call_openai_api(messages: List[Dict[str, Any]], tools=None, response_format=None) -> Dict[str, Any]:
    """Make a direct call to the OpenAI API and return the raw response."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "max_tokens": 1000
    }
    
    if tools:
        data["tools"] = tools
    
    if response_format:
        data["response_format"] = response_format
    
    print(f"\nOpenAI API Request to {url}:\n{json.dumps(data, indent=2)}")
    
    try:
        # For demonstration, use mock responses instead of actual API calls
        if tools and any(tool.get("function", {}).get("name") == "get_weather" for tool in tools):
            return mock_openai_tool_response
        elif response_format and response_format.get("type") == "json_object":
            return mock_openai_structured_response
        elif len(messages) > 1 and any(msg.get("role") == "function" for msg in messages):
            # This is a function response scenario
            return mock_openai_regular_response
        else:
            return mock_openai_regular_response
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        return {}

# Compare regular completion
print("\n=== Comparing Regular Completion ===\n")
regular_messages = [
    {"role": "user", "content": "Hello, how are you today?"}
]

asi_response = call_asi_api(regular_messages)
print(f"\nASI API Response:\n{json.dumps(asi_response, indent=2)}")

openai_response = call_openai_api(regular_messages)
print(f"\nOpenAI API Response:\n{json.dumps(openai_response, indent=2)}")

# Compare tool calling
print("\n=== Comparing Tool Calling ===\n")
tool_messages = [
    {"role": "user", "content": "What's the weather like in San Francisco?"}
]

asi_tool_response = call_asi_api(tool_messages, tools=[weather_tool])
print(f"\nASI API Tool Response:\n{json.dumps(asi_tool_response, indent=2)}")

openai_tool_response = call_openai_api(tool_messages, tools=[weather_tool])
print(f"\nOpenAI API Tool Response:\n{json.dumps(openai_tool_response, indent=2)}")

# Compare structured output
print("\n=== Comparing Structured Output ===\n")
structured_messages = [
    {"role": "user", "content": "Generate a JSON object with name, age, and occupation for John Doe, age 30, Software Engineer"}
]

asi_structured_response = call_asi_api(structured_messages, response_format={"type": "json_object"})
print(f"\nASI API Structured Response:\n{json.dumps(asi_structured_response, indent=2)}")

openai_structured_response = call_openai_api(structured_messages, response_format={"type": "json_object"})
print(f"\nOpenAI API Structured Response:\n{json.dumps(openai_structured_response, indent=2)}")

# Compare function response
print("\n=== Comparing Function Response ===\n")
function_messages = [
    {"role": "user", "content": "What's the weather like in San Francisco?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
            }
        }
    ]},
    {"role": "function", "name": "get_weather", "content": "{\"temperature\": 22, \"conditions\": \"sunny\"}"}
]

asi_function_response = call_asi_api(function_messages)
print(f"\nASI API Function Response:\n{json.dumps(asi_function_response, indent=2)}")

openai_function_response = call_openai_api(function_messages)
print(f"\nOpenAI API Function Response:\n{json.dumps(openai_function_response, indent=2)}")

# Print summary of differences
print("\n=== Summary of Differences Between ASI and OpenAI ===\n")
print("1. ASI API endpoint is /v1/chat/completions at api.asi1.ai")
print("2. ASI requires 'model' field in the request")
print("3. ASI response includes a 'thought' array that OpenAI doesn't have")
print("4. ASI doesn't include 'object' and 'created' fields that OpenAI has")
print("5. The structure of 'choices' array is similar, but ASI might have different finish_reason values")
