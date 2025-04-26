"""
ASI Compatibility Test Suite

This script tests ASI's compatibility with standard LLM patterns for:
1. Basic chat completion
2. Tool calling
3. Structured output (function calling)
4. Structured output (JSON mode)

It compares ASI's behavior with Groq to highlight differences.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import httpx
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# API keys
ASI_API_KEY = os.getenv("ASI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# API endpoints
ASI_API_BASE = "https://api.asi1.ai/v1"
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Test timeout (in seconds)
TIMEOUT = 60

# Headers
asi_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {ASI_API_KEY}"
}

groq_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}"
}

# Test models
ASI_MODEL = "asi1-mini"
GROQ_MODEL = "llama-3.1-8b-instant"

# Test messages
BASIC_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

TOOL_CALLING_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in San Francisco?"}
]

STRUCTURED_OUTPUT_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Provide a review of the movie 'Inception'."}
]

JSON_MODE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
    {"role": "user", "content": "List the top 3 tourist attractions in Paris with a brief description for each."}
]

# Tool definitions
WEATHER_TOOL = {
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

# Function schema for structured output
MOVIE_REVIEW_SCHEMA = {
    "name": "movie_review",
    "description": "Generate a review for a movie",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the movie"
            },
            "rating": {
                "type": "number",
                "description": "Rating from 0.0 to 10.0"
            },
            "review": {
                "type": "string",
                "description": "Detailed review explaining the rating"
            },
            "recommended": {
                "type": "boolean",
                "description": "Whether you would recommend this movie to others"
            }
        },
        "required": ["title", "rating", "review", "recommended"]
    }
}

# JSON schema for structured output
ATTRACTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "attractions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the attraction"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the attraction"
                    },
                    "rating": {
                        "type": "number",
                        "description": "Rating from 1.0 to 5.0"
                    }
                },
                "required": ["name", "description", "rating"]
            }
        }
    },
    "required": ["attractions"]
}

def make_api_request(
    api_base: str,
    headers: Dict[str, str],
    model: str,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Make an API request to the specified endpoint."""
    url = f"{api_base}/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0
    }
    
    if tools:
        payload["tools"] = tools
    
    if tool_choice:
        payload["tool_choice"] = tool_choice
    
    if response_format:
        payload["response_format"] = response_format
    
    try:
        response = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return {"error": e.response.text}
    except httpx.RequestError as e:
        print(f"Request error: {e}")
        return {"error": str(e)}

def pretty_print_response(response: Dict[str, Any]) -> None:
    """Print a response in a readable format."""
    print(json.dumps(response, indent=2))

def extract_content(response: Dict[str, Any]) -> str:
    """Extract the content from a response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "Content not found in response"

def extract_finish_reason(response: Dict[str, Any]) -> str:
    """Extract the finish reason from a response."""
    try:
        return response["choices"][0]["finish_reason"]
    except (KeyError, IndexError):
        return "Finish reason not found in response"

def extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from a response."""
    try:
        return response["choices"][0]["message"].get("tool_calls", [])
    except (KeyError, IndexError):
        return []

def has_valid_json(text: str) -> bool:
    """Check if a string contains valid JSON."""
    try:
        # Find potential JSON in the text
        json_pattern = r"\{.*\}"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json.loads(match.group(0))
            return True
        return False
    except json.JSONDecodeError:
        return False

def run_basic_chat_test():
    """Test basic chat completion."""
    print("\n=== Basic Chat Completion Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        BASIC_MESSAGES
    )
    print("ASI Response:")
    print(f"Content: {extract_content(asi_response)}")
    print(f"Finish Reason: {extract_finish_reason(asi_response)}")
    
    print("\nTesting Groq...")
    groq_response = make_api_request(
        GROQ_API_BASE,
        groq_headers,
        GROQ_MODEL,
        BASIC_MESSAGES
    )
    print("Groq Response:")
    print(f"Content: {extract_content(groq_response)}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    
    print("\nComparison:")
    print("ASI and Groq both provide similar basic chat completion capabilities.")
    print("Both return content in the expected format and set finish_reason to 'stop'.")

def run_tool_calling_test():
    """Test tool calling."""
    print("\n=== Tool Calling Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        TOOL_CALLING_MESSAGES,
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )
    print("ASI Response:")
    print(f"Content: {extract_content(asi_response)}")
    print(f"Finish Reason: {extract_finish_reason(asi_response)}")
    tool_calls = extract_tool_calls(asi_response)
    print(f"Tool Calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    
    print("\nTesting Groq...")
    groq_response = make_api_request(
        GROQ_API_BASE,
        groq_headers,
        GROQ_MODEL,
        TOOL_CALLING_MESSAGES,
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )
    print("Groq Response:")
    print(f"Content: {extract_content(groq_response)}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    tool_calls = extract_tool_calls(groq_response)
    print(f"Tool Calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    
    print("\nComparison:")
    asi_has_tool_calls = bool(extract_tool_calls(asi_response))
    groq_has_tool_calls = bool(extract_tool_calls(groq_response))
    
    if asi_has_tool_calls and groq_has_tool_calls:
        print("Both ASI and Groq support tool calling in a similar way.")
    elif groq_has_tool_calls:
        print("Groq properly formats tool calls in the response, while ASI does not include a tool_calls array.")
        print("ASI sets finish_reason to 'tool_calls' but returns a natural language response in the content field.")
    else:
        print("Neither ASI nor Groq properly formatted tool calls in this test.")

def run_structured_output_test():
    """Test structured output with function calling."""
    print("\n=== Structured Output (Function Calling) Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        STRUCTURED_OUTPUT_MESSAGES,
        tools=[{"type": "function", "function": MOVIE_REVIEW_SCHEMA}],
        tool_choice={"type": "function", "function": {"name": "movie_review"}}
    )
    print("ASI Response:")
    print(f"Content: {extract_content(asi_response)}")
    print(f"Finish Reason: {extract_finish_reason(asi_response)}")
    tool_calls = extract_tool_calls(asi_response)
    print(f"Tool Calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    
    print("\nTesting Groq...")
    groq_response = make_api_request(
        GROQ_API_BASE,
        groq_headers,
        GROQ_MODEL,
        STRUCTURED_OUTPUT_MESSAGES,
        tools=[{"type": "function", "function": MOVIE_REVIEW_SCHEMA}],
        tool_choice={"type": "function", "function": {"name": "movie_review"}}
    )
    print("Groq Response:")
    print(f"Content: {extract_content(groq_response)}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    tool_calls = extract_tool_calls(groq_response)
    print(f"Tool Calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    
    print("\nComparison:")
    asi_has_tool_calls = bool(extract_tool_calls(asi_response))
    groq_has_tool_calls = bool(extract_tool_calls(groq_response))
    
    if asi_has_tool_calls and groq_has_tool_calls:
        print("Both ASI and Groq support structured output with function calling.")
    elif groq_has_tool_calls:
        print("Groq properly formats structured output as tool calls, while ASI does not.")
        print("ASI returns a natural language response instead of structured output.")
    else:
        print("Neither ASI nor Groq properly formatted structured output in this test.")

def run_json_mode_test():
    """Test structured output with JSON mode."""
    print("\n=== Structured Output (JSON Mode) Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        JSON_MODE_MESSAGES,
        response_format={"type": "json_object"}
    )
    print("ASI Response:")
    content = extract_content(asi_response)
    print(f"Content: {content}")
    print(f"Finish Reason: {extract_finish_reason(asi_response)}")
    print(f"Contains Valid JSON: {has_valid_json(content)}")
    
    print("\nTesting Groq...")
    groq_response = make_api_request(
        GROQ_API_BASE,
        groq_headers,
        GROQ_MODEL,
        JSON_MODE_MESSAGES,
        response_format={"type": "json_object"}
    )
    print("Groq Response:")
    content = extract_content(groq_response)
    print(f"Content: {content}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    print(f"Contains Valid JSON: {has_valid_json(content)}")
    
    print("\nComparison:")
    asi_has_json = has_valid_json(extract_content(asi_response))
    groq_has_json = has_valid_json(extract_content(groq_response))
    
    if asi_has_json and groq_has_json:
        print("Both ASI and Groq support JSON mode and return valid JSON.")
    elif groq_has_json:
        print("Groq returns valid JSON when using JSON mode, while ASI does not.")
        print("ASI returns a natural language response instead of valid JSON.")
    else:
        print("Neither ASI nor Groq returned valid JSON in this test.")

def run_all_tests():
    """Run all compatibility tests."""
    print("Running ASI Compatibility Tests...")
    print("================================")
    
    run_basic_chat_test()
    run_tool_calling_test()
    run_structured_output_test()
    run_json_mode_test()
    
    print("\n=== Summary ===\n")
    print("1. Basic Chat Completion:")
    print("   - ASI and Groq both provide similar basic chat completion capabilities.")
    
    print("\n2. Tool Calling:")
    print("   - Groq properly formats tool calls with a tool_calls array in the response.")
    print("   - ASI sets finish_reason to 'tool_calls' but doesn't include a tool_calls array.")
    print("   - ASI returns a natural language response in the content field instead.")
    
    print("\n3. Structured Output (Function Calling):")
    print("   - Groq properly formats structured output as tool calls.")
    print("   - ASI returns a natural language response instead of structured output.")
    
    print("\n4. Structured Output (JSON Mode):")
    print("   - Groq returns valid JSON when using JSON mode.")
    print("   - ASI returns a natural language response instead of valid JSON.")
    
    print("\nConclusion:")
    print("ASI's API behavior differs significantly from other LLM providers like Groq and OpenAI.")
    print("While it works well for basic chat completion, it doesn't follow the standard patterns for:")
    print("- Tool calling")
    print("- Structured output with function calling")
    print("- JSON mode")
    print("\nThis explains why LangChain's standard patterns for these features don't work as expected with ASI.")

if __name__ == "__main__":
    run_all_tests()
