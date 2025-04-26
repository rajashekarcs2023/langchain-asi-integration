"""Test script to examine the ASI integration's tool calling implementation."""

import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_asi import ChatASI

# Load environment variables
load_dotenv()

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

def test_asi_integration_tool_calling():
    """Test the ASI integration's tool calling implementation."""
    print("\n=== Testing ASI Integration Tool Calling ===")
    
    # Initialize ASI model
    asi_model = ChatASI(
        model="asi1-mini",
        temperature=0,
        verbose=True
    )
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant that routes queries to the appropriate agent."),
        HumanMessage(content="What are Apple's key financial risks and how have they changed over the past year?")
    ]
    
    # Test bind_tools
    print("\nTesting bind_tools...")
    bound_model = asi_model.bind_tools(
        tools=[route_tool],
        tool_choice={"type": "function", "function": {"name": "route"}}
    )
    
    # Generate a response
    print("Generating response...")
    response = bound_model.invoke(messages)
    
    # Examine the response
    print("\nResponse Type:", type(response))
    print("Response Content:", response.content)
    print("Response Additional Kwargs:", response.additional_kwargs)
    
    # Check for tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print("\nTool Calls:", response.tool_calls)
    else:
        print("\nNo tool_calls found in the response.")
    
    # Check finish reason
    if hasattr(response, 'response_metadata') and response.response_metadata:
        print("\nResponse Metadata:", response.response_metadata)
        print("Finish Reason:", response.response_metadata.get("finish_reason", "Not specified"))
    
    return response

def test_asi_integration_structured_output():
    """Test the ASI integration's structured output implementation."""
    print("\n=== Testing ASI Integration Structured Output ===")
    
    # Initialize ASI model
    asi_model = ChatASI(
        model="asi1-mini",
        temperature=0,
        verbose=True
    )
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant that routes queries to the appropriate agent."),
        HumanMessage(content="What are Apple's key financial risks and how have they changed over the past year?")
    ]
    
    # Test with_structured_output
    print("\nTesting with_structured_output...")
    structured_model = asi_model.with_structured_output(
        schema=route_tool["function"],
        method="json_mode",
        include_raw=True
    )
    
    # Generate a response
    print("Generating response...")
    response = structured_model.invoke(messages)
    
    # Examine the response
    print("\nResponse Type:", type(response))
    print("Response:", json.dumps(response, indent=2, default=str))
    
    return response

if __name__ == "__main__":
    # Test both implementations
    tool_calling_response = test_asi_integration_tool_calling()
    structured_output_response = test_asi_integration_structured_output()
