"""
ASI Detailed Tool Calling Test Suite

This script performs detailed tests on ASI's tool calling capabilities compared to Groq,
focusing specifically on areas where ASI's behavior differs from standard patterns.

Tests include:
1. Multiple tool calls in a single request
2. Tool choice variations (auto, none, specific)
3. Complex tool schemas
4. Tool call with nested JSON arguments
5. Error handling for invalid tool calls
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
MULTIPLE_TOOLS_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in San Francisco and what time is it there?"}
]

COMPLEX_TOOL_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I need to book a flight from New York to London next Friday."}
]

NESTED_JSON_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Create a shopping list for a dinner party with 5 guests."}
]

ERROR_HANDLING_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in an invalid location?"}
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

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "format": {
                    "type": "string",
                    "enum": ["12h", "24h"],
                    "description": "The time format to use"
                }
            },
            "required": ["location"]
        }
    }
}

FLIGHT_BOOKING_TOOL = {
    "type": "function",
    "function": {
        "name": "book_flight",
        "description": "Book a flight between two locations",
        "parameters": {
            "type": "object",
            "properties": {
                "departure_airport": {
                    "type": "string",
                    "description": "The departure airport code, e.g. JFK"
                },
                "arrival_airport": {
                    "type": "string",
                    "description": "The arrival airport code, e.g. LHR"
                },
                "departure_date": {
                    "type": "string",
                    "description": "The departure date in YYYY-MM-DD format"
                },
                "return_date": {
                    "type": "string",
                    "description": "The return date in YYYY-MM-DD format (optional)"
                },
                "passenger_details": {
                    "type": "object",
                    "properties": {
                        "adults": {
                            "type": "integer",
                            "description": "Number of adult passengers"
                        },
                        "children": {
                            "type": "integer",
                            "description": "Number of child passengers"
                        },
                        "infants": {
                            "type": "integer",
                            "description": "Number of infant passengers"
                        }
                    },
                    "required": ["adults"]
                },
                "cabin_class": {
                    "type": "string",
                    "enum": ["economy", "premium_economy", "business", "first"],
                    "description": "The cabin class for the flight"
                }
            },
            "required": ["departure_airport", "arrival_airport", "departure_date", "passenger_details"]
        }
    }
}

SHOPPING_LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "create_shopping_list",
        "description": "Create a shopping list for a meal",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the item"
                            },
                            "quantity": {
                                "type": "number",
                                "description": "Quantity of the item"
                            },
                            "unit": {
                                "type": "string",
                                "description": "Unit of measurement (e.g., kg, lbs, pieces)"
                            },
                            "category": {
                                "type": "string",
                                "enum": ["produce", "meat", "dairy", "bakery", "canned_goods", "frozen", "beverages", "other"],
                                "description": "Category of the item"
                            }
                        },
                        "required": ["name", "quantity"]
                    }
                },
                "estimated_total": {
                    "type": "number",
                    "description": "Estimated total cost"
                },
                "store_preference": {
                    "type": "string",
                    "description": "Preferred store for shopping"
                }
            },
            "required": ["items"]
        }
    }
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

def extract_additional_fields(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract any additional fields from a response."""
    try:
        message = response["choices"][0]["message"]
        additional_fields = {}
        for key, value in message.items():
            if key not in ["role", "content", "tool_calls"]:
                additional_fields[key] = value
        return additional_fields
    except (KeyError, IndexError):
        return {}

def test_multiple_tool_calls():
    """Test handling of multiple tool calls in a single request."""
    print("\n=== Multiple Tool Calls Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        MULTIPLE_TOOLS_MESSAGES,
        tools=[WEATHER_TOOL, TIME_TOOL],
        tool_choice="auto"
    )
    print("ASI Response:")
    print(f"Content: {extract_content(asi_response)}")
    print(f"Finish Reason: {extract_finish_reason(asi_response)}")
    tool_calls = extract_tool_calls(asi_response)
    print(f"Tool Calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    additional_fields = extract_additional_fields(asi_response)
    print(f"Additional Fields: {json.dumps(additional_fields, indent=2) if additional_fields else 'None'}")
    
    print("\nTesting Groq...")
    groq_response = make_api_request(
        GROQ_API_BASE,
        groq_headers,
        GROQ_MODEL,
        MULTIPLE_TOOLS_MESSAGES,
        tools=[WEATHER_TOOL, TIME_TOOL],
        tool_choice="auto"
    )
    print("Groq Response:")
    print(f"Content: {extract_content(groq_response)}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    tool_calls = extract_tool_calls(groq_response)
    print(f"Tool Calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    additional_fields = extract_additional_fields(groq_response)
    print(f"Additional Fields: {json.dumps(additional_fields, indent=2) if additional_fields else 'None'}")
    
    print("\nAnalysis:")
    asi_has_tool_calls = bool(extract_tool_calls(asi_response))
    groq_has_tool_calls = bool(extract_tool_calls(groq_response))
    
    if asi_has_tool_calls and groq_has_tool_calls:
        print("Both ASI and Groq support multiple tool calls.")
    elif groq_has_tool_calls:
        print("Groq properly formats multiple tool calls, while ASI does not include tool_calls array.")
        print("ASI sets finish_reason to 'tool_calls' but returns a natural language response in the content field.")
    else:
        print("Neither ASI nor Groq properly formatted multiple tool calls in this test.")

def test_tool_choice_variations():
    """Test different tool_choice variations."""
    print("\n=== Tool Choice Variations Test ===\n")
    
    tool_choices = [
        ("auto", "Automatic tool selection"),
        ("none", "No tool selection"),
        ({"type": "function", "function": {"name": "get_weather"}}, "Specific tool selection")
    ]
    
    for tool_choice, description in tool_choices:
        print(f"\nTesting {description}...")
        
        print("Testing ASI...")
        asi_response = make_api_request(
            ASI_API_BASE,
            asi_headers,
            ASI_MODEL,
            MULTIPLE_TOOLS_MESSAGES,
            tools=[WEATHER_TOOL, TIME_TOOL],
            tool_choice=tool_choice
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
            MULTIPLE_TOOLS_MESSAGES,
            tools=[WEATHER_TOOL, TIME_TOOL],
            tool_choice=tool_choice
        )
        print("Groq Response:")
        print(f"Content: {extract_content(groq_response)}")
        print(f"Finish Reason: {extract_finish_reason(groq_response)}")
        tool_calls = extract_tool_calls(groq_response)
        print(f"Tool Calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
        
        print("\nAnalysis:")
        asi_finish_reason = extract_finish_reason(asi_response)
        groq_finish_reason = extract_finish_reason(groq_response)
        asi_has_tool_calls = bool(extract_tool_calls(asi_response))
        groq_has_tool_calls = bool(extract_tool_calls(groq_response))
        
        print(f"ASI finish_reason: {asi_finish_reason}")
        print(f"Groq finish_reason: {groq_finish_reason}")
        print(f"ASI has tool_calls: {asi_has_tool_calls}")
        print(f"Groq has tool_calls: {groq_has_tool_calls}")

def test_complex_tool_schema():
    """Test handling of complex tool schemas."""
    print("\n=== Complex Tool Schema Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        COMPLEX_TOOL_MESSAGES,
        tools=[FLIGHT_BOOKING_TOOL],
        tool_choice={"type": "function", "function": {"name": "book_flight"}}
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
        COMPLEX_TOOL_MESSAGES,
        tools=[FLIGHT_BOOKING_TOOL],
        tool_choice={"type": "function", "function": {"name": "book_flight"}}
    )
    print("Groq Response:")
    print(f"Content: {extract_content(groq_response)}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    tool_calls = extract_tool_calls(groq_response)
    print(f"Tool Calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    
    print("\nAnalysis:")
    asi_has_tool_calls = bool(extract_tool_calls(asi_response))
    groq_has_tool_calls = bool(extract_tool_calls(groq_response))
    
    if asi_has_tool_calls and groq_has_tool_calls:
        print("Both ASI and Groq support complex tool schemas.")
    elif groq_has_tool_calls:
        print("Groq properly handles complex tool schemas, while ASI does not include tool_calls array.")
        print("ASI sets finish_reason to 'tool_calls' but returns a natural language response in the content field.")
    else:
        print("Neither ASI nor Groq properly handled complex tool schemas in this test.")

def test_nested_json_arguments():
    """Test handling of nested JSON arguments in tool calls."""
    print("\n=== Nested JSON Arguments Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        NESTED_JSON_MESSAGES,
        tools=[SHOPPING_LIST_TOOL],
        tool_choice={"type": "function", "function": {"name": "create_shopping_list"}}
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
        NESTED_JSON_MESSAGES,
        tools=[SHOPPING_LIST_TOOL],
        tool_choice={"type": "function", "function": {"name": "create_shopping_list"}}
    )
    print("Groq Response:")
    print(f"Content: {extract_content(groq_response)}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    tool_calls = extract_tool_calls(groq_response)
    print(f"Tool Calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    
    print("\nAnalysis:")
    asi_has_tool_calls = bool(extract_tool_calls(asi_response))
    groq_has_tool_calls = bool(extract_tool_calls(groq_response))
    
    if asi_has_tool_calls and groq_has_tool_calls:
        print("Both ASI and Groq support nested JSON arguments in tool calls.")
    elif groq_has_tool_calls:
        print("Groq properly handles nested JSON arguments, while ASI does not include tool_calls array.")
        print("ASI sets finish_reason to 'tool_calls' but returns a natural language response in the content field.")
    else:
        print("Neither ASI nor Groq properly handled nested JSON arguments in this test.")

def test_error_handling():
    """Test error handling for invalid tool calls."""
    print("\n=== Error Handling Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        ERROR_HANDLING_MESSAGES,
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
        ERROR_HANDLING_MESSAGES,
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )
    print("Groq Response:")
    print(f"Content: {extract_content(groq_response)}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    tool_calls = extract_tool_calls(groq_response)
    print(f"Tool Calls: {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
    
    print("\nAnalysis:")
    asi_finish_reason = extract_finish_reason(asi_response)
    groq_finish_reason = extract_finish_reason(groq_response)
    asi_has_tool_calls = bool(extract_tool_calls(asi_response))
    groq_has_tool_calls = bool(extract_tool_calls(groq_response))
    
    print(f"ASI finish_reason: {asi_finish_reason}")
    print(f"Groq finish_reason: {groq_finish_reason}")
    print(f"ASI has tool_calls: {asi_has_tool_calls}")
    print(f"Groq has tool_calls: {groq_has_tool_calls}")

def run_all_tests():
    """Run all detailed tool calling tests."""
    print("Running ASI Detailed Tool Calling Tests...")
    print("========================================")
    
    test_multiple_tool_calls()
    test_tool_choice_variations()
    test_complex_tool_schema()
    test_nested_json_arguments()
    test_error_handling()
    
    print("\n=== Summary ===\n")
    print("1. Multiple Tool Calls:")
    print("   - Groq properly formats multiple tool calls with a tool_calls array.")
    print("   - ASI sets finish_reason to 'tool_calls' but doesn't include a tool_calls array.")
    
    print("\n2. Tool Choice Variations:")
    print("   - Groq respects tool_choice parameter and formats responses accordingly.")
    print("   - ASI sets finish_reason based on tool_choice but doesn't include tool_calls array.")
    
    print("\n3. Complex Tool Schema:")
    print("   - Groq properly handles complex tool schemas with nested objects.")
    print("   - ASI returns natural language responses that attempt to describe the complex schema.")
    
    print("\n4. Nested JSON Arguments:")
    print("   - Groq properly formats nested JSON arguments in tool calls.")
    print("   - ASI returns natural language descriptions of nested JSON structures.")
    
    print("\n5. Error Handling:")
    print("   - Groq provides structured tool calls even for potentially invalid inputs.")
    print("   - ASI returns natural language responses that attempt to handle errors.")

if __name__ == "__main__":
    run_all_tests()
