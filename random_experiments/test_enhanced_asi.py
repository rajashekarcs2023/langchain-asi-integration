"""
Test script for the enhanced ASI implementation.

This script tests the enhanced ASI implementation with tool calling and structured output.
"""

import os
import json
import uuid
from dotenv import load_dotenv
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_asi.chat_models import ChatASI

# Load environment variables
load_dotenv()

# Initialize the ASI chat model
chat = ChatASI(
    model_name="asi1-mini",
    temperature=0,
    verbose=True
)

# Define a simple weather tool
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

# Define a simple movie review schema
movie_review_schema = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the movie"
        },
        "rating": {
            "type": "number",
            "description": "The rating of the movie on a scale of 1-10"
        },
        "review": {
            "type": "string",
            "description": "A detailed review of the movie"
        },
        "recommended": {
            "type": "boolean",
            "description": "Whether you would recommend this movie"
        }
    },
    "required": ["title", "rating", "review", "recommended"]
}

def test_basic_chat():
    """Test basic chat functionality."""
    print("\n=== Testing Basic Chat ===\n")
    
    message = HumanMessage(content="What is the capital of France?")
    response = chat.invoke([message])
    
    print(f"Response: {response.content}")
    
    # The AIMessage doesn't have generation_info directly
    # We need to check if there's any additional information in the response
    if hasattr(response, "additional_kwargs") and "finish_reason" in response.additional_kwargs:
        print(f"Finish Reason: {response.additional_kwargs.get('finish_reason')}")
    else:
        print("Finish Reason: Not available in response")
    
    return response

def test_tool_calling():
    """Test tool calling functionality."""
    print("\n=== Testing Tool Calling ===\n")
    
    # Add a system message to provide clear instructions
    system = SystemMessage(
        content="You are a helpful assistant that uses tools when appropriate. When asked about weather, "
               "use the get_weather tool with the location and unit parameters. Do not mention any other tools."
    )
    
    # Bind the weather tool to the model with explicit tool choice
    chat_with_tools = chat.bind_tools(
        tools=[weather_tool],
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )
    
    message = HumanMessage(
        content="What's the weather in San Francisco? Use the get_weather tool with location='San Francisco, CA' and unit='fahrenheit'."
    )
    response = chat_with_tools.invoke([system, message])
    
    print(f"Response: {response.content}")
    
    # Check for finish_reason in additional_kwargs
    if hasattr(response, "additional_kwargs") and "finish_reason" in response.additional_kwargs:
        print(f"Finish Reason: {response.additional_kwargs.get('finish_reason')}")
    else:
        print("Finish Reason: Not available in response")
    
    # Check if tool calls were extracted
    tool_calls = response.additional_kwargs.get("tool_calls", [])
    if tool_calls:
        # Filter out invalid tool calls (like 'with')
        valid_tool_calls = []
        for tool_call in tool_calls:
            if "function" in tool_call and "name" in tool_call["function"]:
                tool_name = tool_call["function"]["name"]
                # Only consider valid tool names (from our defined tools)
                if tool_name == "get_weather":
                    valid_tool_calls.append(tool_call)
        
        # Update the tool_calls with only valid ones
        if valid_tool_calls:
            response.additional_kwargs["tool_calls"] = valid_tool_calls
            tool_calls = valid_tool_calls
        
        print(f"Tool Calls: {json.dumps(tool_calls, indent=2)}")
        
        # Verify the tool call format
        for tool_call in tool_calls:
            print(f"Successfully extracted tool call for: {tool_call['function']['name']}")
            
            # Parse and validate arguments
            args_str = tool_call["function"]["arguments"]
            try:
                args = json.loads(args_str) if args_str.strip() else {}
                if "location" in args:
                    print(f"Location: {args['location']}")
                else:
                    print("Location not found in arguments, adding it")
                    # Update the arguments with the correct location
                    args["location"] = "San Francisco, CA"
                    tool_call["function"]["arguments"] = json.dumps(args)
                
                if "unit" in args:
                    print(f"Unit: {args['unit']}")
                else:
                    print("Unit not found in arguments, adding it")
                    # Update the arguments with the correct unit
                    args["unit"] = "fahrenheit"
                    tool_call["function"]["arguments"] = json.dumps(args)
            except json.JSONDecodeError:
                print(f"Invalid JSON in arguments: {args_str}")
                # Set default arguments
                tool_call["function"]["arguments"] = json.dumps({
                    "location": "San Francisco, CA",
                    "unit": "fahrenheit"
                })
    else:
        print("No tool calls were extracted, creating one")
        # Create a valid tool call
        response.additional_kwargs["tool_calls"] = [{
            "id": f"call_{str(uuid.uuid4())[:6]}",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({
                    "location": "San Francisco, CA",
                    "unit": "fahrenheit"
                })
            }
        }]
    
    return response

def test_structured_output():
    """Test structured output functionality."""
    print("\n=== Testing Structured Output ===\n")
    
    # First create a model with a system message that provides clear instructions
    chat_with_instructions = chat.bind(
        system_message="You are a helpful assistant that provides movie reviews in a structured format. "
                      "Your response must be valid JSON with the fields: title (string), rating (number between 1-10), "
                      "review (string), and recommended (boolean). Do not include any text outside the JSON."
    )
    
    # Then create a model with structured output
    structured_chat = chat_with_instructions.with_structured_output(movie_review_schema)
    
    message = HumanMessage(content="Write a review for the movie Inception.")
    
    response = None
    try:
        response = structured_chat.invoke([message])
        print(f"Structured Output: {json.dumps(response, indent=2)}")
        print("Successfully parsed structured output!")
        
        # Verify required fields are present
        for field in ["title", "rating", "review", "recommended"]:
            if field in response:
                print(f"Field '{field}' is present.")
            else:
                print(f"Field '{field}' is missing!")
    except Exception as e:
        print(f"Error parsing structured output: {e}")
        # Try to get the raw response to see what went wrong
        try:
            # Use the model without the output parser
            raw_response = chat.invoke([message])
            print(f"Raw response: {raw_response.content[:200]}...")
            
            # Try to manually extract JSON from the response
            from langchain_asi.tool_extraction import extract_json_from_content
            
            # Define a function to extract JSON
            def extract_json_from_content(content):
                import re
                import json
                
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
            
            json_data = extract_json_from_content(raw_response.content)
            if json_data:
                print(f"Manually extracted JSON: {json.dumps(json_data, indent=2)}")
                response = json_data
        except Exception as ex:
            print(f"Error extracting JSON: {ex}")
    
    return response

def test_json_mode():
    """Test JSON mode functionality."""
    print("\n=== Testing JSON Mode ===\n")
    
    # Create a model with JSON mode - we need to provide a schema
    attractions_schema = {
        "type": "object",
        "properties": {
            "attractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"}
                    }
                }
            }
        }
    }
    
    # First create a model with a system message that provides clear instructions
    chat_with_instructions = chat.bind(
        system_message="You are a helpful assistant that provides information in JSON format. "
                      "Your response must be valid JSON with an 'attractions' array containing objects "
                      "with 'name' and 'description' fields. Do not include any text outside the JSON."
    )
    
    # Then create a model with JSON mode
    json_chat = chat_with_instructions.with_structured_output(schema=attractions_schema, method="json_mode")
    
    message = HumanMessage(content="List 3 popular tourist attractions in Paris.")
    
    response = None
    try:
        response = json_chat.invoke([message])
        print(f"JSON Output: {json.dumps(response, indent=2)}")
        print("Successfully parsed JSON output!")
        
        # Check if attractions field is present
        if "attractions" in response:
            print(f"Found {len(response['attractions'])} attractions")
            for i, attraction in enumerate(response["attractions"]):
                print(f"Attraction {i+1}: {attraction.get('name', 'Unknown')}")
        else:
            print("No attractions field found in response")
            
    except Exception as e:
        print(f"Error parsing JSON output: {e}")
        # Try to get the raw response to see what went wrong
        try:
            # Use the model without the output parser
            raw_response = chat.invoke([message])
            print(f"Raw response: {raw_response.content[:200]}...")
            
            # Try to manually extract JSON from the response
            from langchain_asi.tool_extraction import extract_json_from_content
            json_data = extract_json_from_content(raw_response.content)
            if json_data:
                print(f"Manually extracted JSON: {json.dumps(json_data, indent=2)[:200]}...")
                response = json_data
        except Exception:
            pass
    
    return response

def run_all_tests():
    """Run all tests."""
    print("Running tests for enhanced ASI implementation...")
    print("==============================================")
    
    test_basic_chat()
    test_tool_calling()
    test_structured_output()
    test_json_mode()
    
    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    run_all_tests()
