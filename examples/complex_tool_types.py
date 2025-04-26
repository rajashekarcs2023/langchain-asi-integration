"""Example demonstrating complex tool types with ASI integration."""

import os
import sys
from typing import Dict, List, Optional, Union, Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_asi import ChatASI

# Load environment variables
load_dotenv()

# Define a Pydantic model for a complex tool
class SearchQuery(BaseModel):
    """Search query parameters."""
    query: str = Field(description="The search query string")
    filters: Optional[Dict[str, str]] = Field(
        default=None, 
        description="Optional filters to apply to the search"
    )
    limit: int = Field(
        default=5, 
        description="Maximum number of results to return"
    )

class SearchResult(BaseModel):
    """Search result item."""
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    snippet: str = Field(description="Text snippet from the search result")

# Define a complex tool using a Pydantic model
def search_web(query: SearchQuery) -> List[SearchResult]:
    """Search the web for information.
    
    Args:
        query: The search query parameters
        
    Returns:
        A list of search results
    """
    # This is a mock implementation
    print(f"Searching for: {query.query}")
    if query.filters:
        print(f"With filters: {query.filters}")
    print(f"Limit: {query.limit}")
    
    # Return mock results
    return [
        SearchResult(
            title=f"Result {i} for {query.query}",
            url=f"https://example.com/result{i}",
            snippet=f"This is a snippet for result {i} related to {query.query}"
        )
        for i in range(1, min(query.limit + 1, 6))
    ]

# Define a tool with nested parameters
@tool
def book_flight(
    departure: str, 
    destination: str, 
    date: str, 
    passengers: int = 1,
    class_type: str = "economy",
    preferences: Optional[Dict[str, str]] = None
) -> str:
    """Book a flight with the given parameters.
    
    Args:
        departure: Departure city or airport code
        destination: Destination city or airport code
        date: Date of departure in YYYY-MM-DD format
        passengers: Number of passengers
        class_type: Class type (economy, business, first)
        preferences: Optional preferences like meal, seat, etc.
        
    Returns:
        Booking confirmation details
    """
    # This is a mock implementation
    print(f"Booking flight from {departure} to {destination} on {date}")
    print(f"For {passengers} passengers in {class_type} class")
    if preferences:
        print(f"With preferences: {preferences}")
    
    # Return mock booking confirmation
    return f"Flight booked successfully! Confirmation code: XYZ123. {departure} to {destination} on {date} for {passengers} passengers in {class_type} class."

# Create a custom tool class
class ImageGenerationTool(BaseTool):
    name = "generate_image"
    description = "Generate an image based on a text description"
    
    def _run(self, prompt: str, style: str = "realistic", size: str = "1024x1024") -> str:
        """Run the image generation tool.
        
        Args:
            prompt: Text description of the image to generate
            style: Style of the image (realistic, cartoon, abstract, etc.)
            size: Size of the image in pixels (width x height)
            
        Returns:
            URL to the generated image
        """
        # This is a mock implementation
        print(f"Generating {style} image of size {size} for prompt: {prompt}")
        
        # Return mock image URL
        return f"https://example.com/images/{prompt.replace(' ', '_')}_{style}_{size}.jpg"
    
    def _arun(self, prompt: str, style: str = "realistic", size: str = "1024x1024") -> str:
        """Run the image generation tool asynchronously."""
        # For simplicity, we're just calling the synchronous version
        return self._run(prompt, style, size)

def test_complex_tools():
    """Test complex tool types with ASI."""
    print("Testing complex tool types with ASI...")
    
    # Initialize the chat model
    chat = ChatASI(
        model_name="asi1-mini",
        temperature=0,
        verbose=True
    )
    
    # Test 1: Pydantic model-based tool
    print("\n=== Test 1: Pydantic model-based tool ===")
    chat_with_search = chat.bind_tools(
        tools=[search_web],
        tool_choice="any"  # Force the model to use a tool
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant that can search the web. Always use the search_web tool when asked to search for information."),
        HumanMessage(content="Search for information about artificial intelligence and limit to 3 results.")
    ]
    
    try:
        response = chat_with_search.invoke(messages)
        print("\nResponse:")
        print(f"Content: {response.content}")
        
        # Check if tool calls were made
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        print(f"\nTool calls: {len(tool_calls)}")
        for i, tool_call in enumerate(tool_calls):
            print(f"\nTool call {i+1}:")
            print(f"Type: {tool_call.get('type')}")
            print(f"Function: {tool_call.get('function', {}).get('name')}")
            print(f"Arguments: {tool_call.get('function', {}).get('arguments')}")
    except Exception as e:
        print(f"\nError: {e}")
    
    # Test 2: Complex parameter tool
    print("\n=== Test 2: Complex parameter tool ===")
    chat_with_booking = chat.bind_tools(
        tools=[book_flight],
        tool_choice="any"  # Force the model to use a tool
    )
    
    messages = [
        SystemMessage(content="You are a travel assistant that can book flights. Always use the book_flight tool when asked to book a flight."),
        HumanMessage(content="Book a flight from New York to London on 2025-05-15 for 2 passengers in business class. I prefer window seats and vegetarian meals.")
    ]
    
    try:
        response = chat_with_booking.invoke(messages)
        print("\nResponse:")
        print(f"Content: {response.content}")
        
        # Check if tool calls were made
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        print(f"\nTool calls: {len(tool_calls)}")
        for i, tool_call in enumerate(tool_calls):
            print(f"\nTool call {i+1}:")
            print(f"Type: {tool_call.get('type')}")
            print(f"Function: {tool_call.get('function', {}).get('name')}")
            print(f"Arguments: {tool_call.get('function', {}).get('arguments')}")
    except Exception as e:
        print(f"\nError: {e}")
    
    # Test 3: Custom tool class
    print("\n=== Test 3: Custom tool class ===")
    image_tool = ImageGenerationTool()
    chat_with_image = chat.bind_tools(
        tools=[image_tool],
        tool_choice="any"  # Force the model to use a tool
    )
    
    messages = [
        SystemMessage(content="You are a creative assistant that can generate images. Always use the generate_image tool when asked to create an image."),
        HumanMessage(content="Generate an abstract image of a sunset over the ocean at 512x512 resolution.")
    ]
    
    try:
        response = chat_with_image.invoke(messages)
        print("\nResponse:")
        print(f"Content: {response.content}")
        
        # Check if tool calls were made
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        print(f"\nTool calls: {len(tool_calls)}")
        for i, tool_call in enumerate(tool_calls):
            print(f"\nTool call {i+1}:")
            print(f"Type: {tool_call.get('type')}")
            print(f"Function: {tool_call.get('function', {}).get('name')}")
            print(f"Arguments: {tool_call.get('function', {}).get('arguments')}")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_complex_tools()
    sys.exit(0)
