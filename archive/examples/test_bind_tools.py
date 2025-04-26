"""
Test script for the bind_tools method in the ChatASI class.

This script demonstrates how to use the bind_tools method to enable tool calling
with the ASI language model through the LangChain integration.
"""

import os
import json
from typing import List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_asi.chat_models import ChatASI

# Initialize the ChatASI model
asi_api_key = os.environ.get("ASI_API_KEY")
if not asi_api_key:
    raise ValueError("ASI_API_KEY environment variable is not set")

# Create a ChatASI instance
chat_asi = ChatASI(
    model_name="asi1-mini",
    temperature=0,
    max_tokens=1000,
    verbose=True
)

# Define a simple tool using the @tool decorator
@tool
def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The unit of temperature, either "celsius" or "fahrenheit"
        
    Returns:
        The current weather conditions and temperature
    """
    # This is a mock implementation
    weather_data = {
        "San Francisco, CA": {"conditions": "sunny", "temp": 72},
        "New York, NY": {"conditions": "cloudy", "temp": 65},
        "Chicago, IL": {"conditions": "windy", "temp": 58},
        "Miami, FL": {"conditions": "rainy", "temp": 80},
    }
    
    # Default to a generic response for locations not in our mock data
    data = weather_data.get(location, {"conditions": "clear", "temp": 70})
    
    # Convert temperature if necessary
    if unit.lower() == "celsius":
        data["temp"] = round((data["temp"] - 32) * 5/9)
    
    return f"The current weather in {location} is {data['conditions']} with a temperature of {data['temp']}°{'C' if unit.lower() == 'celsius' else 'F'}"

# Define a Pydantic model for structured output
class MovieReview(BaseModel):
    """A structured movie review with title, rating, and comments."""
    title: str = Field(description="The title of the movie being reviewed")
    rating: float = Field(description="Rating from 0.0 to 10.0")
    review: str = Field(description="Detailed review explaining the rating")
    recommended: bool = Field(description="Whether you would recommend this movie to others")

def test_tool_calling():
    """Test the bind_tools method with a simple tool."""
    print("\n=== Testing Tool Calling ===\n")
    
    # Bind the weather tool to the model
    model_with_tools = chat_asi.bind_tools([get_current_weather])
    
    # Create a message asking about the weather
    messages = [HumanMessage(content="What's the weather like in San Francisco?")]
    
    # Invoke the model with the message
    response = model_with_tools.invoke(messages)
    
    print(f"Response: {response.content}")
    print(f"Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'None'}")
    
    return response

def test_structured_output():
    """Test the with_structured_output method."""
    print("\n=== Testing Structured Output ===\n")
    
    # Create a model with structured output
    structured_model = chat_asi.with_structured_output(
        MovieReview,
        method="function_calling"
    )
    
    # Create a message asking for a movie review
    message = [
        SystemMessage(content="You are a movie critic who provides detailed reviews."),
        HumanMessage(content="Write a review for the movie 'The Matrix'")
    ]
    
    # Invoke the model with the message
    response = structured_model.invoke(message)
    
    print(f"Response type: {type(response)}")
    print(f"Response: {response}")
    
    return response

def test_json_mode():
    """Test the with_structured_output method using JSON mode."""
    print("\n=== Testing JSON Mode ===\n")
    
    # Create a model with structured output using JSON mode
    json_model = chat_asi.with_structured_output(
        MovieReview,
        method="json_mode"
    )
    
    # Create a message asking for a movie review with explicit JSON instructions
    message = [
        SystemMessage(content="You are a movie critic who provides detailed reviews. Your responses must be in valid JSON format that matches the MovieReview schema exactly."),
        HumanMessage(content="Write a review for the movie 'Inception'. Return your response as a JSON object with fields: title (string), rating (number between 0-10), review (string), and recommended (boolean).")
    ]
    
    try:
        # Invoke the model with the message
        response = json_model.invoke(message)
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        return response
    except Exception as e:
        print(f"Error with JSON mode: {e}")
        print("\nFalling back to function calling method instead...\n")
        
        # Try with function calling method instead
        structured_model = chat_asi.with_structured_output(
            MovieReview,
            method="function_calling"
        )
        response = structured_model.invoke(message)
        print(f"Function calling response type: {type(response)}")
        print(f"Function calling response: {response}")
        return response

def test_multiple_tools():
    """Test binding multiple tools to the model."""
    print("\n=== Testing Multiple Tools ===\n")
    
    # Define another tool
    @tool
    def get_movie_info(title: str) -> str:
        """Get information about a movie.
        
        Args:
            title: The title of the movie
            
        Returns:
            Information about the movie including release year and director
        """
        # Mock data
        movie_data = {
            "The Matrix": {"year": 1999, "director": "The Wachowskis"},
            "Inception": {"year": 2010, "director": "Christopher Nolan"},
            "Titanic": {"year": 1997, "director": "James Cameron"},
        }
        
        data = movie_data.get(title, {"year": "unknown", "director": "unknown"})
        return f"{title} was released in {data['year']} and directed by {data['director']}."
    
    # Bind multiple tools to the model
    model_with_tools = chat_asi.bind_tools([get_current_weather, get_movie_info])
    
    # Create a message that could use either tool
    messages = [HumanMessage(content="I'm curious about the movie Inception and also wondering about the weather in Chicago.")]
    
    # Invoke the model with the message
    response = model_with_tools.invoke(messages)
    
    print(f"Response: {response.content}")
    print(f"Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'None'}")
    
    return response

if __name__ == "__main__":
    # Run all tests
    test_tool_calling()
    test_structured_output()
    test_json_mode()
    test_multiple_tools()
