"""
ASI Detailed Structured Output Test Suite

This script performs detailed tests on ASI's structured output capabilities compared to Groq,
focusing specifically on areas where ASI's behavior differs from standard patterns.

Tests include:
1. Basic JSON mode
2. Complex nested JSON structures
3. Function calling with simple schemas
4. Function calling with complex schemas
5. JSON extraction from different response formats
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
BASIC_JSON_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "List 3 popular programming languages as a JSON array."}
]

COMPLEX_JSON_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Create a JSON object representing a company with departments, employees, and projects."}
]

SIMPLE_FUNCTION_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

COMPLEX_FUNCTION_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Analyze the sentiment of the following text: 'I absolutely loved the movie! The acting was superb, though the ending was a bit confusing.'"}
]

JSON_EXTRACTION_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Generate a recipe for chocolate chip cookies in JSON format."}
]

# Function definitions
CAPITAL_FUNCTION = {
    "type": "function",
    "function": {
        "name": "get_capital",
        "description": "Get the capital city of a country",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "The country name"
                },
                "capital": {
                    "type": "string",
                    "description": "The capital city name"
                }
            },
            "required": ["country", "capital"]
        }
    }
}

SENTIMENT_FUNCTION = {
    "type": "function",
    "function": {
        "name": "analyze_sentiment",
        "description": "Analyze the sentiment of a text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze"
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "mixed"],
                    "description": "The overall sentiment of the text"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0 and 1"
                },
                "aspects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "aspect": {
                                "type": "string",
                                "description": "The aspect being evaluated"
                            },
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"],
                                "description": "The sentiment for this aspect"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Explanation for the sentiment"
                            }
                        },
                        "required": ["aspect", "sentiment"]
                    },
                    "description": "Sentiment breakdown by aspects"
                }
            },
            "required": ["text", "sentiment", "confidence"]
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

def extract_json_from_content(content: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from content, handling various formats."""
    # Try to extract JSON from markdown code blocks
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    json_blocks = re.findall(json_block_pattern, content)
    
    for block in json_blocks:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to extract JSON directly
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON with relaxed parsing
    try:
        # Remove any non-JSON text before the first {
        if "{" in content:
            content = content[content.find("{"):]
        elif "[" in content:
            content = content[content.find("["):]
        
        # Remove any non-JSON text after the last } or ]
        if "}" in content:
            content = content[:content.rfind("}") + 1]
        elif "]" in content:
            content = content[:content.rfind("]") + 1]
        
        return json.loads(content.strip())
    except (json.JSONDecodeError, ValueError):
        return None

def is_valid_json(content: str) -> bool:
    """Check if content contains valid JSON."""
    return extract_json_from_content(content) is not None

def test_basic_json_mode():
    """Test basic JSON mode."""
    print("\n=== Basic JSON Mode Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        BASIC_JSON_MESSAGES,
        response_format={"type": "json_object"}
    )
    asi_content = extract_content(asi_response)
    print("ASI Response:")
    print(f"Content: {asi_content}")
    print(f"Finish Reason: {extract_finish_reason(asi_response)}")
    print(f"Contains Valid JSON: {is_valid_json(asi_content)}")
    
    if is_valid_json(asi_content):
        print(f"Extracted JSON: {json.dumps(extract_json_from_content(asi_content), indent=2)}")
    
    print("\nTesting Groq...")
    groq_response = make_api_request(
        GROQ_API_BASE,
        groq_headers,
        GROQ_MODEL,
        BASIC_JSON_MESSAGES,
        response_format={"type": "json_object"}
    )
    groq_content = extract_content(groq_response)
    print("Groq Response:")
    print(f"Content: {groq_content}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    print(f"Contains Valid JSON: {is_valid_json(groq_content)}")
    
    if is_valid_json(groq_content):
        print(f"Extracted JSON: {json.dumps(extract_json_from_content(groq_content), indent=2)}")
    
    print("\nAnalysis:")
    asi_valid_json = is_valid_json(asi_content)
    groq_valid_json = is_valid_json(groq_content)
    
    if asi_valid_json and groq_valid_json:
        print("Both ASI and Groq return valid JSON in basic JSON mode.")
    elif groq_valid_json:
        print("Groq returns valid JSON, while ASI does not.")
        print("ASI returns a natural language response or JSON wrapped in text.")
    else:
        print("Neither ASI nor Groq returned valid JSON in this test.")

def test_complex_json_structures():
    """Test complex nested JSON structures."""
    print("\n=== Complex JSON Structures Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        COMPLEX_JSON_MESSAGES,
        response_format={"type": "json_object"}
    )
    asi_content = extract_content(asi_response)
    print("ASI Response:")
    print(f"Content: {asi_content}")
    print(f"Finish Reason: {extract_finish_reason(asi_response)}")
    print(f"Contains Valid JSON: {is_valid_json(asi_content)}")
    
    if is_valid_json(asi_content):
        print(f"Extracted JSON: {json.dumps(extract_json_from_content(asi_content), indent=2)}")
    
    print("\nTesting Groq...")
    groq_response = make_api_request(
        GROQ_API_BASE,
        groq_headers,
        GROQ_MODEL,
        COMPLEX_JSON_MESSAGES,
        response_format={"type": "json_object"}
    )
    groq_content = extract_content(groq_response)
    print("Groq Response:")
    print(f"Content: {groq_content}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    print(f"Contains Valid JSON: {is_valid_json(groq_content)}")
    
    if is_valid_json(groq_content):
        print(f"Extracted JSON: {json.dumps(extract_json_from_content(groq_content), indent=2)}")
    
    print("\nAnalysis:")
    asi_valid_json = is_valid_json(asi_content)
    groq_valid_json = is_valid_json(groq_content)
    
    if asi_valid_json and groq_valid_json:
        print("Both ASI and Groq handle complex nested JSON structures.")
    elif groq_valid_json:
        print("Groq properly handles complex nested JSON structures, while ASI does not.")
        print("ASI returns a natural language response or JSON wrapped in text.")
    else:
        print("Neither ASI nor Groq properly handled complex nested JSON structures in this test.")

def test_simple_function_calling():
    """Test function calling with simple schemas."""
    print("\n=== Simple Function Calling Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        SIMPLE_FUNCTION_MESSAGES,
        tools=[CAPITAL_FUNCTION],
        tool_choice={"type": "function", "function": {"name": "get_capital"}}
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
        SIMPLE_FUNCTION_MESSAGES,
        tools=[CAPITAL_FUNCTION],
        tool_choice={"type": "function", "function": {"name": "get_capital"}}
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
        print("Both ASI and Groq support simple function calling.")
    elif groq_has_tool_calls:
        print("Groq properly formats simple function calls, while ASI does not include tool_calls array.")
        print("ASI sets finish_reason to 'tool_calls' but returns a natural language response in the content field.")
    else:
        print("Neither ASI nor Groq properly formatted simple function calls in this test.")

def test_complex_function_calling():
    """Test function calling with complex schemas."""
    print("\n=== Complex Function Calling Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        COMPLEX_FUNCTION_MESSAGES,
        tools=[SENTIMENT_FUNCTION],
        tool_choice={"type": "function", "function": {"name": "analyze_sentiment"}}
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
        COMPLEX_FUNCTION_MESSAGES,
        tools=[SENTIMENT_FUNCTION],
        tool_choice={"type": "function", "function": {"name": "analyze_sentiment"}}
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
        print("Both ASI and Groq support complex function calling.")
    elif groq_has_tool_calls:
        print("Groq properly formats complex function calls, while ASI does not include tool_calls array.")
        print("ASI sets finish_reason to 'tool_calls' but returns a natural language response in the content field.")
    else:
        print("Neither ASI nor Groq properly formatted complex function calls in this test.")

def test_json_extraction():
    """Test JSON extraction from different response formats."""
    print("\n=== JSON Extraction Test ===\n")
    
    print("Testing ASI...")
    asi_response = make_api_request(
        ASI_API_BASE,
        asi_headers,
        ASI_MODEL,
        JSON_EXTRACTION_MESSAGES,
        response_format={"type": "json_object"}
    )
    asi_content = extract_content(asi_response)
    print("ASI Response:")
    print(f"Content: {asi_content}")
    print(f"Finish Reason: {extract_finish_reason(asi_response)}")
    
    # Test different extraction methods
    print("\nJSON Extraction Methods:")
    
    # Method 1: Direct JSON parsing
    try:
        json_data = json.loads(asi_content)
        print("Method 1 (Direct JSON parsing): Success")
        print(f"Extracted: {json.dumps(json_data, indent=2)[:100]}...")
    except json.JSONDecodeError:
        print("Method 1 (Direct JSON parsing): Failed")
    
    # Method 2: Markdown code block extraction
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    json_blocks = re.findall(json_block_pattern, asi_content)
    
    if json_blocks:
        try:
            json_data = json.loads(json_blocks[0].strip())
            print("Method 2 (Markdown code block extraction): Success")
            print(f"Extracted: {json.dumps(json_data, indent=2)[:100]}...")
        except json.JSONDecodeError:
            print("Method 2 (Markdown code block extraction): Failed")
    else:
        print("Method 2 (Markdown code block extraction): No code blocks found")
    
    # Method 3: Relaxed JSON parsing
    try:
        # Remove any non-JSON text before the first {
        if "{" in asi_content:
            content = asi_content[asi_content.find("{"):]
        elif "[" in asi_content:
            content = asi_content[asi_content.find("["):]
        else:
            content = asi_content
        
        # Remove any non-JSON text after the last } or ]
        if "}" in content:
            content = content[:content.rfind("}") + 1]
        elif "]" in content:
            content = content[:content.rfind("]") + 1]
        
        json_data = json.loads(content.strip())
        print("Method 3 (Relaxed JSON parsing): Success")
        print(f"Extracted: {json.dumps(json_data, indent=2)[:100]}...")
    except (json.JSONDecodeError, ValueError):
        print("Method 3 (Relaxed JSON parsing): Failed")
    
    print("\nTesting Groq...")
    groq_response = make_api_request(
        GROQ_API_BASE,
        groq_headers,
        GROQ_MODEL,
        JSON_EXTRACTION_MESSAGES,
        response_format={"type": "json_object"}
    )
    groq_content = extract_content(groq_response)
    print("Groq Response:")
    print(f"Content: {groq_content}")
    print(f"Finish Reason: {extract_finish_reason(groq_response)}")
    
    # Test direct JSON parsing for Groq
    try:
        json_data = json.loads(groq_content)
        print("Direct JSON parsing: Success")
        print(f"Extracted: {json.dumps(json_data, indent=2)[:100]}...")
    except json.JSONDecodeError:
        print("Direct JSON parsing: Failed")
    
    print("\nAnalysis:")
    asi_valid_json = is_valid_json(asi_content)
    groq_valid_json = is_valid_json(groq_content)
    
    if asi_valid_json and groq_valid_json:
        print("Both ASI and Groq return extractable JSON.")
    elif groq_valid_json:
        print("Groq returns directly parseable JSON, while ASI requires special extraction methods.")
        print("ASI often wraps JSON in text or code blocks, requiring more robust parsing.")
    else:
        print("Neither ASI nor Groq returned extractable JSON in this test.")

def run_all_tests():
    """Run all detailed structured output tests."""
    print("Running ASI Detailed Structured Output Tests...")
    print("============================================")
    
    test_basic_json_mode()
    test_complex_json_structures()
    test_simple_function_calling()
    test_complex_function_calling()
    test_json_extraction()
    
    print("\n=== Summary ===\n")
    print("1. Basic JSON Mode:")
    print("   - Groq returns valid JSON directly parseable with standard methods.")
    print("   - ASI often returns JSON wrapped in text or code blocks, requiring special extraction.")
    
    print("\n2. Complex JSON Structures:")
    print("   - Groq handles complex nested JSON structures properly.")
    print("   - ASI may struggle with complex structures or wrap them in explanatory text.")
    
    print("\n3. Simple Function Calling:")
    print("   - Groq properly formats simple function calls with a tool_calls array.")
    print("   - ASI sets finish_reason to 'tool_calls' but doesn't include a tool_calls array.")
    
    print("\n4. Complex Function Calling:")
    print("   - Groq properly formats complex function calls with nested schemas.")
    print("   - ASI returns natural language responses that attempt to describe the complex schema.")
    
    print("\n5. JSON Extraction:")
    print("   - Groq JSON can be extracted with standard json.loads().")
    print("   - ASI requires more robust extraction methods to handle various response formats.")
    print("   - Successful extraction from ASI often requires handling markdown code blocks or text wrapping.")

if __name__ == "__main__":
    run_all_tests()
