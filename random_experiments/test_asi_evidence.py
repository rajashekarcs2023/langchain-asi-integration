"""
Test script to gather evidence of ASI API behavior for compatibility reporting.
This script captures raw API responses for different use cases.
"""

import os
import json
import uuid
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_asi.chat_models import ChatASI
from langchain_core.tools import Tool
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Set up API base URL
ASI_API_BASE = os.getenv("ASI_API_BASE", "https://api.asi1.ai/v1")
print(f"Using ASI API base URL: {ASI_API_BASE}")

# Initialize the model
asi = ChatASI(
    asi_api_key=os.getenv("ASI_API_KEY"),
    asi_api_base=ASI_API_BASE,
    model_name="asi1-mini",
    verbose=True  # Enable verbose to see full API requests and responses
)

# Create a directory for evidence
EVIDENCE_DIR = "asi_evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

def save_evidence(test_name: str, data: Dict[str, Any]) -> None:
    """Save test evidence to a JSON file."""
    file_path = os.path.join(EVIDENCE_DIR, f"{test_name}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Evidence saved to {file_path}")

def test_basic_chat() -> None:
    """Test basic chat functionality and capture the response."""
    print("\n=== Testing Basic Chat ===\n")
    
    # Create a simple query
    messages = [HumanMessage(content="What is the capital of France?")]
    
    # Make the API call and capture the response
    # Convert LangChain message types to ASI expected roles
    asi_messages = []
    for m in messages:
        if m.type == "human":
            asi_messages.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            asi_messages.append({"role": "assistant", "content": m.content})
        else:
            asi_messages.append({"role": m.type, "content": m.content})
    
    response = asi.completion_with_retry(
        messages=asi_messages
    )
    
    # Extract the content
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Response: {content}")
    
    # Check if finish_reason is present
    finish_reason = response.get("choices", [{}])[0].get("finish_reason")
    if finish_reason:
        print(f"Finish Reason: {finish_reason}")
    else:
        print("Finish Reason: Not available in response")
    
    # Save evidence
    save_evidence("basic_chat", {
        "test_case": "Basic Chat",
        "query": "What is the capital of France?",
        "raw_response": response,
        "issues_identified": [
            "Additional fields" if "thought" in str(response) else None,
            "Think tags" if "<think>" in str(response) else None,
            "Missing finish_reason" if not finish_reason else None
        ]
    })

def test_tool_calling() -> None:
    """Test tool calling functionality and capture the response."""
    print("\n=== Testing Tool Calling ===\n")
    
    # Create a query that should trigger tool use
    messages = [
        SystemMessage(content="You are a helpful assistant that uses tools when appropriate."),
        HumanMessage(content="What's the weather like in San Francisco, CA? Use Fahrenheit.")
    ]
    
    # Make the API call with tool_choice="auto"
    # Convert LangChain message types to ASI expected roles
    asi_messages = []
    for m in messages:
        if m.type == "human":
            asi_messages.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            asi_messages.append({"role": "assistant", "content": m.content})
        else:
            asi_messages.append({"role": m.type, "content": m.content})
    
    response = asi.completion_with_retry(
        messages=asi_messages,
        tools=[{
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
        }],
        tool_choice="auto"
    )
    
    # Extract the content
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Response: {content}")
    
    # Check if finish_reason is present
    finish_reason = response.get("choices", [{}])[0].get("finish_reason")
    if finish_reason:
        print(f"Finish Reason: {finish_reason}")
    else:
        print("Finish Reason: Not available in response")
    
    # Check if tool_calls is present
    message = response.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls", [])
    
    # Identify issues
    issues = []
    if finish_reason == "tool_calls" and not tool_calls:
        issues.append("Missing tool_calls array when finish_reason is tool_calls")
        print("ASI indicated tool usage but no tool_calls array is present")
    
    if content and ("get_weather" in content.lower() or "weather" in content.lower()):
        issues.append("Tool intent expressed in natural language content instead of structured tool_calls")
    
    # Save evidence
    save_evidence("tool_calling", {
        "test_case": "Tool Calling",
        "query": "What's the weather like in San Francisco, CA? Use Fahrenheit.",
        "raw_response": response,
        "issues_identified": issues
    })

def test_structured_output() -> None:
    """Test structured output functionality and capture the response."""
    print("\n=== Testing Structured Output ===\n")
    
    # Create a query that should return structured data
    messages = [
        SystemMessage(content="You are a helpful movie critic."),
        HumanMessage(content="Write a short review of the movie Inception.")
    ]
    
    # Make the API call with function calling approach
    # Convert LangChain message types to ASI expected roles
    asi_messages = []
    for m in messages:
        if m.type == "human":
            asi_messages.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            asi_messages.append({"role": "assistant", "content": m.content})
        else:
            asi_messages.append({"role": m.type, "content": m.content})
    
    response = asi.completion_with_retry(
        messages=asi_messages,
        functions=[{
            "name": "MovieReview",
            "description": "Movie review with rating and recommendation",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the movie"
                    },
                    "rating": {
                        "type": "number",
                        "description": "Rating from 0-10"
                    },
                    "review": {
                        "type": "string",
                        "description": "Brief review of the movie"
                    },
                    "recommended": {
                        "type": "boolean",
                        "description": "Whether you recommend this movie"
                    }
                },
                "required": ["title", "rating", "review", "recommended"]
            }
        }],
        function_call={"name": "MovieReview"}
    )
    
    # Extract the content
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Structured Output Response: {content}")
    
    # Try to extract function call if present
    message = response.get("choices", [{}])[0].get("message", {})
    function_call = message.get("function_call", {})
    
    # Identify issues
    issues = []
    if not function_call:
        issues.append("Missing function_call in response")
    
    if function_call and "arguments" in function_call:
        try:
            args = json.loads(function_call["arguments"])
            missing_fields = []
            for field in ["title", "rating", "review", "recommended"]:
                if field not in args:
                    missing_fields.append(field)
            
            if missing_fields:
                issues.append(f"Missing required fields in function arguments: {', '.join(missing_fields)}")
        except json.JSONDecodeError:
            issues.append("Invalid JSON in function arguments")
    
    # Save evidence
    save_evidence("structured_output", {
        "test_case": "Structured Output",
        "query": "Write a short review of the movie Inception.",
        "raw_response": response,
        "issues_identified": issues
    })

def test_json_mode() -> None:
    """Test JSON mode functionality and capture the response."""
    print("\n=== Testing JSON Mode ===\n")
    
    # Create a query that should return JSON
    messages = [
        SystemMessage(content="You are a helpful assistant that always responds with valid JSON."),
        HumanMessage(content="List 3 popular tourist attractions in Paris.")
    ]
    
    # Make the API call with response_format={"type": "json_object"}
    # Convert LangChain message types to ASI expected roles
    asi_messages = []
    for m in messages:
        if m.type == "human":
            asi_messages.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            asi_messages.append({"role": "assistant", "content": m.content})
        else:
            asi_messages.append({"role": m.type, "content": m.content})
    
    response = asi.completion_with_retry(
        messages=asi_messages,
        response_format={"type": "json_object"}
    )
    
    # Extract the content
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"JSON Mode Response: {content}")
    
    # Try to parse as JSON
    is_valid_json = True
    try:
        json.loads(content)
    except json.JSONDecodeError:
        is_valid_json = False
    
    # Identify issues
    issues = []
    if not is_valid_json:
        issues.append("Response is not valid JSON despite response_format=json_object")
        print("Error: Response is not valid JSON")
    
    # Save evidence
    save_evidence("json_mode", {
        "test_case": "JSON Mode",
        "query": "List 3 popular tourist attractions in Paris.",
        "raw_response": response,
        "issues_identified": issues,
        "is_valid_json": is_valid_json
    })

def test_parallel_tool_calls() -> None:
    """Test parallel tool calling functionality and capture the response."""
    print("\n=== Testing Parallel Tool Calls ===\n")
    
    # Define multiple tools
    tools = [
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
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_population",
                "description": "Get the population of a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    # Create a query that should trigger multiple tool uses
    messages = [
        SystemMessage(content="You are a helpful assistant that uses tools when appropriate."),
        HumanMessage(content="I'm planning a trip to San Francisco and New York. What's the weather and population in both cities?")
    ]
    
    # Make the API call with parallel tool calling
    # Convert LangChain message types to ASI expected roles
    asi_messages = []
    for m in messages:
        if m.type == "human":
            asi_messages.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            asi_messages.append({"role": "assistant", "content": m.content})
        else:
            asi_messages.append({"role": m.type, "content": m.content})
    
    response = asi.completion_with_retry(
        messages=asi_messages,
        tools=tools,
        tool_choice="auto"
    )
    
    # Extract the content
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Response: {content}")
    
    # Check if tool_calls is present
    message = response.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls", [])
    
    # Identify issues
    issues = []
    if not tool_calls and "finish_reason" in response.get("choices", [{}])[0] and response["choices"][0]["finish_reason"] == "tool_calls":
        issues.append("Missing tool_calls array when finish_reason is tool_calls")
    
    if tool_calls and len(tool_calls) < 2:
        issues.append("Failed to make parallel tool calls when multiple tools were appropriate")
    
    # Save evidence
    save_evidence("parallel_tool_calls", {
        "test_case": "Parallel Tool Calls",
        "query": "I'm planning a trip to San Francisco and New York. What's the weather and population in both cities?",
        "raw_response": response,
        "issues_identified": issues,
        "tool_calls_count": len(tool_calls) if tool_calls else 0
    })

def run_all_tests() -> None:
    """Run all tests and compile a summary report."""
    print("Running tests to gather evidence of ASI API behavior...")
    print("==============================================")
    
    test_basic_chat()
    test_tool_calling()
    test_structured_output()
    test_json_mode()
    test_parallel_tool_calls()
    
    # Compile summary report
    summary = {
        "test_timestamp": str(uuid.uuid4()),
        "api_base": ASI_API_BASE,
        "model": "asi1-mini",
        "tests_run": ["basic_chat", "tool_calling", "structured_output", "json_mode", "parallel_tool_calls"],
        "summary": "Evidence gathering complete. See individual JSON files for detailed results."
    }
    
    save_evidence("summary", summary)
    print("\n=== All Tests Completed ===")
    print("Evidence files saved in the 'asi_evidence' directory")

if __name__ == "__main__":
    run_all_tests()
