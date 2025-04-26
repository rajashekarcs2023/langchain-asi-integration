"""Test script to compare Groq and ASI tool calling implementations."""

import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_asi import ChatASI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

# Load environment variables
load_dotenv()

# Define a simple tool schema
route_schema = {
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
    },
}

def test_groq_tool_calling():
    """Test Groq's tool calling implementation."""
    print("\n=== Testing Groq Tool Calling ===")
    
    # Initialize Groq model
    groq_model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        verbose=True
    )
    
    # Create a chain with tool calling
    groq_chain = groq_model.bind_tools(tools=[route_schema], tool_choice="route")
    
    # Add the parser
    parser_chain = groq_chain | JsonOutputFunctionsParser()
    
    # Test query
    query = "What are Apple's key financial risks and how have they changed over the past year?"
    messages = [HumanMessage(content=query)]
    
    print(f"\nSending query to Groq: {query}")
    
    # First, get the raw response without the parser
    raw_response = groq_chain.invoke(messages)
    print("\nRaw Groq Response:")
    print(f"Type: {type(raw_response)}")
    print(f"Content: {raw_response}")
    if hasattr(raw_response, 'additional_kwargs'):
        print(f"Additional kwargs: {raw_response.additional_kwargs}")
    if hasattr(raw_response, 'tool_calls'):
        print(f"Tool calls: {raw_response.tool_calls}")
    
    # Now get the parsed response
    try:
        parsed_response = parser_chain.invoke(messages)
        print("\nParsed Groq Response:")
        print(f"Type: {type(parsed_response)}")
        print(f"Content: {parsed_response}")
    except Exception as e:
        print(f"\nError parsing Groq response: {str(e)}")

def test_asi_tool_calling():
    """Test ASI's tool calling implementation."""
    print("\n=== Testing ASI Tool Calling ===")
    
    # Initialize ASI model
    asi_model = ChatASI(
        model="asi1-mini",
        temperature=0,
        verbose=True
    )
    
    # Create a chain with tool calling
    asi_chain = asi_model.bind_tools(tools=[route_schema], tool_choice="route")
    
    # Add the parser
    parser_chain = asi_chain | JsonOutputFunctionsParser()
    
    # Test query
    query = "What are Apple's key financial risks and how have they changed over the past year?"
    messages = [HumanMessage(content=query)]
    
    print(f"\nSending query to ASI: {query}")
    
    # First, get the raw response without the parser
    raw_response = asi_chain.invoke(messages)
    print("\nRaw ASI Response:")
    print(f"Type: {type(raw_response)}")
    print(f"Content: {raw_response}")
    if hasattr(raw_response, 'additional_kwargs'):
        print(f"Additional kwargs: {raw_response.additional_kwargs}")
    if hasattr(raw_response, 'tool_calls'):
        print(f"Tool calls: {raw_response.tool_calls}")
    
    # Now get the parsed response
    try:
        parsed_response = parser_chain.invoke(messages)
        print("\nParsed ASI Response:")
        print(f"Type: {type(parsed_response)}")
        print(f"Content: {parsed_response}")
    except Exception as e:
        print(f"\nError parsing ASI response: {str(e)}")

if __name__ == "__main__":
    # Test both implementations
    test_groq_tool_calling()
    test_asi_tool_calling()
