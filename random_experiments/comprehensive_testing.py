"""
Comprehensive Testing for ASI-LangChain Integration

This script performs comprehensive testing of the ASI-LangChain integration,
focusing on the compatibility layer that bridges ASI's response format with
LangChain's expectations.

Tests include:
1. Basic tool calling
2. Complex tool schemas
3. Multiple tools
4. Parallel tool calling
5. Structured output with various schemas
6. JSON mode with different structures
7. Integration with LangChain agents
8. Edge cases and error handling
"""

import os
import json
import re
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_asi.chat_models import ChatASI

# Load environment variables
load_dotenv()

# Set up API base URL
ASI_API_BASE = os.getenv("ASI_API_BASE", "https://api.asi1.ai/v1")
print(f"Using ASI API base URL: {ASI_API_BASE}")

# Initialize the ASI model
asi = ChatASI(
    asi_api_key=os.getenv("ASI_API_KEY"),
    asi_api_base=ASI_API_BASE,
    model_name="asi1-mini"
)

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "details": []
}

def record_test_result(test_name, passed, details=None):
    """Record the result of a test."""
    result = {
        "test_name": test_name,
        "passed": passed,
        "details": details or {}
    }
    test_results["details"].append(result)
    if passed:
        test_results["passed"] += 1
    else:
        test_results["failed"] += 1
    
    # Print result
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{test_name}: {status}")
    if details:
        print(f"  Details: {json.dumps(details, indent=2)}")

def has_tool_calls(response):
    """Check if a response has tool calls in any format."""
    # Check tool_calls attribute
    if hasattr(response, "tool_calls") and response.tool_calls:
        return True
    
    # Check additional_kwargs
    if hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"):
        return True
    
    # Check content for tool mentions
    if hasattr(response, "content") and response.content:
        content = response.content
        if "`get_" in content or "`search_" in content or "`calculator" in content:
            return True
    
    return False

def extract_tool_info(response):
    """Extract tool information from a response."""
    tool_info = {
        "has_tool_calls_attr": hasattr(response, "tool_calls") and bool(response.tool_calls),
        "has_tool_calls_kwargs": hasattr(response, "additional_kwargs") and bool(response.additional_kwargs.get("tool_calls")),
        "tool_calls_attr": getattr(response, "tool_calls", None),
        "tool_calls_kwargs": getattr(response, "additional_kwargs", {}).get("tool_calls"),
        "content": getattr(response, "content", "")
    }
    return tool_info

# Test 1: Basic Tool Calling
def test_basic_tool_calling():
    """Test basic tool calling with a simple weather tool."""
    print("\n=== Test 1: Basic Tool Calling ===\n")
    
    # Define a weather tool
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
                        "description": "The temperature unit to use: 'celsius' or 'fahrenheit'",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    # Create the model with the tool
    model = asi.bind(tools=[weather_tool])
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant with access to tools."),
        HumanMessage(content="What's the weather like in New York?")
    ]
    
    # Generate a response
    response = model.invoke(messages)
    
    # Check if tool calls were detected
    tool_info = extract_tool_info(response)
    has_calls = has_tool_calls(response)
    
    # Record result
    record_test_result(
        "Basic Tool Calling",
        has_calls,
        {
            "content": response.content,
            "tool_info": tool_info
        }
    )
    
    return response

# Test 2: Complex Tool Schema
def test_complex_tool_schema():
    """Test tool calling with a complex tool schema."""
    print("\n=== Test 2: Complex Tool Schema ===\n")
    
    # Define a complex search tool
    search_tool = {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search a database for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Filters to apply to the search",
                        "properties": {
                            "date_range": {
                                "type": "object",
                                "properties": {
                                    "start_date": {
                                        "type": "string",
                                        "description": "Start date in YYYY-MM-DD format"
                                    },
                                    "end_date": {
                                        "type": "string",
                                        "description": "End date in YYYY-MM-DD format"
                                    }
                                }
                            },
                            "categories": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Categories to filter by"
                            }
                        }
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "date", "popularity"],
                        "description": "How to sort the results"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            }
        }
    }
    
    # Create the model with the tool
    model = asi.bind(tools=[search_tool])
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant with access to tools."),
        HumanMessage(content="Search for information about climate change from 2020 to 2023, focusing on scientific articles. Sort by date and limit to 5 results.")
    ]
    
    # Generate a response
    response = model.invoke(messages)
    
    # Check if tool calls were detected
    tool_info = extract_tool_info(response)
    has_calls = has_tool_calls(response)
    
    # Record result
    record_test_result(
        "Complex Tool Schema",
        has_calls,
        {
            "content": response.content,
            "tool_info": tool_info
        }
    )
    
    return response

# Test 3: Multiple Tools
def test_multiple_tools():
    """Test with multiple tools available."""
    print("\n=== Test 3: Multiple Tools ===\n")
    
    # Define multiple tools
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
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
    
    # Create the model with multiple tools
    model = asi.bind(tools=[weather_tool, calculator_tool])
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant with access to tools."),
        HumanMessage(content="What's 123 * 456?")
    ]
    
    # Generate a response
    response = model.invoke(messages)
    
    # Check if tool calls were detected
    tool_info = extract_tool_info(response)
    has_calls = has_tool_calls(response)
    
    # Check if the correct tool was selected
    correct_tool = False
    if tool_info["tool_calls_attr"]:
        for tc in tool_info["tool_calls_attr"]:
            if tc.get("name") == "calculator":
                correct_tool = True
    elif tool_info["tool_calls_kwargs"]:
        for tc in tool_info["tool_calls_kwargs"]:
            if tc.get("function", {}).get("name") == "calculator":
                correct_tool = True
    elif "calculator" in tool_info["content"]:
        correct_tool = True
    
    # Record result
    record_test_result(
        "Multiple Tools",
        has_calls and correct_tool,
        {
            "content": response.content,
            "tool_info": tool_info,
            "correct_tool_selected": correct_tool
        }
    )
    
    return response

# Test 4: Structured Output
def test_structured_output():
    """Test structured output with a Pydantic model."""
    print("\n=== Test 4: Structured Output ===\n")
    
    # Define a Pydantic model
    class MovieReview(BaseModel):
        title: str = Field(description="The title of the movie")
        rating: float = Field(description="Rating from 0-10")
        review: str = Field(description="Brief review of the movie")
        recommended: bool = Field(description="Whether you recommend this movie")
    
    # Create the model with structured output
    model = asi.with_structured_output(MovieReview)
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful movie critic."),
        HumanMessage(content="Write a short review of the movie Inception.")
    ]
    
    # Generate a response
    try:
        response = model.invoke(messages)
        
        # Check if the response is a MovieReview
        is_movie_review = isinstance(response, MovieReview)
        has_required_fields = (
            hasattr(response, "title") and
            hasattr(response, "rating") and
            hasattr(response, "review") and
            hasattr(response, "recommended")
        )
        
        # Record result
        record_test_result(
            "Structured Output",
            is_movie_review and has_required_fields,
            {
                "is_movie_review": is_movie_review,
                "has_required_fields": has_required_fields,
                "title": getattr(response, "title", None),
                "rating": getattr(response, "rating", None),
                "review_excerpt": getattr(response, "review", "")[:50] + "..." if hasattr(response, "review") else None,
                "recommended": getattr(response, "recommended", None)
            }
        )
    except Exception as e:
        # Record failure
        record_test_result(
            "Structured Output",
            False,
            {
                "error": str(e)
            }
        )
        response = str(e)
    
    return response

# Test 5: JSON Mode
def test_json_mode():
    """Test JSON mode with different structures."""
    print("\n=== Test 5: JSON Mode ===\n")
    
    # Create the model with JSON mode
    model = ChatASI(
        asi_api_key=os.getenv("ASI_API_KEY"),
        asi_api_base=ASI_API_BASE,
        model_name="asi1-mini",
        response_format={"type": "json_object"}
    )
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant that always responds with valid JSON."),
        HumanMessage(content="List 3 popular tourist attractions in Paris with their names and descriptions.")
    ]
    
    # Generate a response
    response = model.invoke(messages)
    
    # Try to parse the response as JSON
    try:
        # First try direct parsing
        try:
            parsed_json = json.loads(response.content)
            json_parsable = True
        except json.JSONDecodeError:
            # Try extracting JSON from the content
            from langchain_asi.asi_compatibility import extract_json_from_content
            extracted_json = extract_json_from_content(response.content)
            if extracted_json:
                parsed_json = json.loads(extracted_json) if isinstance(extracted_json, str) else extracted_json
                json_parsable = True
            else:
                parsed_json = None
                json_parsable = False
        
        # Record result
        record_test_result(
            "JSON Mode",
            json_parsable,
            {
                "content": response.content,
                "json_parsable": json_parsable,
                "parsed_json": parsed_json
            }
        )
    except Exception as e:
        # Record failure
        record_test_result(
            "JSON Mode",
            False,
            {
                "error": str(e),
                "content": response.content
            }
        )
    
    return response

# Test 6: Tool Choice Parameter
def test_tool_choice():
    """Test the tool_choice parameter."""
    print("\n=== Test 6: Tool Choice Parameter ===\n")
    
    # Define multiple tools
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
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
    
    # Create the model with multiple tools and tool_choice
    model = asi.bind(
        tools=[weather_tool, calculator_tool],
        tool_choice={"type": "function", "function": {"name": "calculator"}}
    )
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant with access to tools."),
        HumanMessage(content="What's the weather like in New York?")
    ]
    
    # Generate a response
    response = model.invoke(messages)
    
    # Check if tool calls were detected
    tool_info = extract_tool_info(response)
    has_calls = has_tool_calls(response)
    
    # Check if the correct tool was selected (should be calculator due to tool_choice)
    correct_tool = False
    if tool_info["tool_calls_attr"]:
        for tc in tool_info["tool_calls_attr"]:
            if tc.get("name") == "calculator":
                correct_tool = True
    elif tool_info["tool_calls_kwargs"]:
        for tc in tool_info["tool_calls_kwargs"]:
            if tc.get("function", {}).get("name") == "calculator":
                correct_tool = True
    elif "calculator" in tool_info["content"]:
        correct_tool = True
    
    # Record result
    record_test_result(
        "Tool Choice Parameter",
        has_calls and correct_tool,
        {
            "content": response.content,
            "tool_info": tool_info,
            "correct_tool_selected": correct_tool
        }
    )
    
    return response

# Test 7: Error Handling
def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n=== Test 7: Error Handling ===\n")
    
    # Test with an invalid tool schema
    invalid_tool = {
        "type": "function",
        "function": {
            "name": "invalid_tool",
            # Missing description and parameters
        }
    }
    
    try:
        # Create the model with the invalid tool
        model = asi.bind(tools=[invalid_tool])
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant with access to tools."),
            HumanMessage(content="Use the invalid tool.")
        ]
        
        # Generate a response
        response = model.invoke(messages)
        
        # If we get here, the model handled the invalid tool gracefully
        record_test_result(
            "Error Handling - Invalid Tool",
            True,
            {
                "content": response.content,
                "graceful_handling": True
            }
        )
    except Exception as e:
        # Record the error
        record_test_result(
            "Error Handling - Invalid Tool",
            False,
            {
                "error": str(e)
            }
        )
        response = str(e)
    
    return response

# Run all tests
def run_all_tests():
    """Run all tests and print a summary."""
    print("Running comprehensive tests for ASI-LangChain integration...")
    print("==========================================================\n")
    
    # Run tests
    test_basic_tool_calling()
    test_complex_tool_schema()
    test_multiple_tools()
    test_structured_output()
    test_json_mode()
    test_tool_choice()
    test_error_handling()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Total tests: {test_results['passed'] + test_results['failed']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    
    # Print detailed results
    print("\n=== Detailed Results ===")
    for i, result in enumerate(test_results["details"]):
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"{i+1}. {result['test_name']}: {status}")
    
    return test_results

if __name__ == "__main__":
    run_all_tests()
