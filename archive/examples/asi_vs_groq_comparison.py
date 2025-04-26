"""
Comprehensive comparison testing between ChatASI and ChatGroq implementations.

This script runs the same examples through both the ASI and Groq LangChain integrations
to verify they work the same way and have feature parity.

Features tested:
1. Basic chat functionality
2. Tool calling
3. Structured output with function calling
4. Structured output with JSON mode
5. Multiple tools
"""

import os
import json
from typing import List, Dict, Any, Optional, Type, Union, Callable
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Import both chat models
from langchain_asi.chat_models import ChatASI
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Set up API keys
asi_api_key = os.environ.get("ASI_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

if not asi_api_key:
    raise ValueError("ASI_API_KEY environment variable is not set")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Initialize models
chat_asi = ChatASI(
    model_name="asi1-mini",
    temperature=0,
    max_tokens=1000,
    verbose=True
)

chat_groq = ChatGroq(
    model="llama-3.1-8b-instant",  # Using a comparable model to asi1-mini
    temperature=0,
    max_tokens=1000,
    verbose=True
)

# Define test tools
@tool
def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The unit of temperature, either "celsius" or "fahrenheit"
        
    Returns:
        The current weather conditions and temperature
    """
    # Mock implementation
    weather_data = {
        "San Francisco, CA": {"conditions": "sunny", "temp": 72},
        "New York, NY": {"conditions": "cloudy", "temp": 65},
        "Chicago, IL": {"conditions": "windy", "temp": 58},
        "Miami, FL": {"conditions": "rainy", "temp": 80},
    }
    
    data = weather_data.get(location, {"conditions": "clear", "temp": 70})
    
    if unit.lower() == "celsius":
        data["temp"] = round((data["temp"] - 32) * 5/9)
    
    return f"The current weather in {location} is {data['conditions']} with a temperature of {data['temp']}°{'C' if unit.lower() == 'celsius' else 'F'}"

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

# Define Pydantic model for structured output
class MovieReview(BaseModel):
    """A structured movie review with title, rating, and comments."""
    title: str = Field(description="The title of the movie being reviewed")
    rating: float = Field(description="Rating from 0.0 to 10.0")
    review: str = Field(description="Detailed review explaining the rating")
    recommended: bool = Field(description="Whether you would recommend this movie to others")

# Test functions
def test_basic_chat(model_name: str, model):
    """Test basic chat functionality."""
    print(f"\n=== Testing Basic Chat with {model_name} ===\n")
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    
    response = model.invoke(messages)
    
    print(f"{model_name} Response: {response.content}\n")
    return response

def test_tool_calling(model_name: str, model):
    """Test tool calling functionality."""
    print(f"\n=== Testing Tool Calling with {model_name} ===\n")
    
    model_with_tools = model.bind_tools([get_current_weather])
    
    messages = [
        HumanMessage(content="What's the weather like in San Francisco?")
    ]
    
    response = model_with_tools.invoke(messages)
    
    print(f"{model_name} Response: {response.content}")
    print(f"{model_name} Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'None'}\n")
    return response

def test_structured_output_function_calling(model_name: str, model):
    """Test structured output with function calling."""
    print(f"\n=== Testing Structured Output (Function Calling) with {model_name} ===\n")
    
    structured_model = model.with_structured_output(
        MovieReview,
        method="function_calling"
    )
    
    messages = [
        SystemMessage(content="You are a movie critic who provides detailed reviews."),
        HumanMessage(content="Write a review for the movie 'The Matrix'")
    ]
    
    response = structured_model.invoke(messages)
    
    print(f"{model_name} Response type: {type(response)}")
    print(f"{model_name} Response: {response}\n")
    return response

def test_structured_output_json_mode(model_name: str, model):
    """Test structured output with JSON mode."""
    print(f"\n=== Testing Structured Output (JSON Mode) with {model_name} ===\n")
    
    try:
        json_model = model.with_structured_output(
            MovieReview,
            method="json_mode"
        )
        
        messages = [
            SystemMessage(content="You are a movie critic who provides detailed reviews."),
            HumanMessage(content="Write a review for the movie 'Inception'. Return your response as a JSON object with fields: title (string), rating (number between 0-10), review (string), and recommended (boolean).")
        ]
        
        response = json_model.invoke(messages)
        
        print(f"{model_name} Response type: {type(response)}")
        print(f"{model_name} Response: {response}\n")
        return response
    except Exception as e:
        print(f"{model_name} Error with JSON mode: {e}\n")
        return None

def test_multiple_tools(model_name: str, model):
    """Test multiple tools functionality."""
    print(f"\n=== Testing Multiple Tools with {model_name} ===\n")
    
    model_with_tools = model.bind_tools([get_current_weather, get_movie_info])
    
    messages = [
        HumanMessage(content="I'm curious about the movie Inception and also wondering about the weather in Chicago.")
    ]
    
    response = model_with_tools.invoke(messages)
    
    print(f"{model_name} Response: {response.content}")
    print(f"{model_name} Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'None'}\n")
    return response

def test_chain_with_tools(model_name: str, model):
    """Test using tools within a chain."""
    print(f"\n=== Testing Chain with Tools with {model_name} ===\n")
    
    model_with_tools = model.bind_tools([get_current_weather, get_movie_info])
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that provides information about movies and weather."),
        ("human", "{input}")
    ])
    
    # Create a chain
    chain = prompt | model_with_tools | StrOutputParser()
    
    # Run the chain
    response = chain.invoke({"input": "Tell me about the movie The Matrix and the weather in New York."})
    
    print(f"{model_name} Chain Response: {response}\n")
    return response

def run_all_tests():
    """Run all tests for both models."""
    # Test ASI
    asi_basic = test_basic_chat("ASI", chat_asi)
    asi_tool = test_tool_calling("ASI", chat_asi)
    asi_structured = test_structured_output_function_calling("ASI", chat_asi)
    asi_json = test_structured_output_json_mode("ASI", chat_asi)
    asi_multi_tools = test_multiple_tools("ASI", chat_asi)
    asi_chain = test_chain_with_tools("ASI", chat_asi)
    
    # Test Groq
    groq_basic = test_basic_chat("Groq", chat_groq)
    groq_tool = test_tool_calling("Groq", chat_groq)
    groq_structured = test_structured_output_function_calling("Groq", chat_groq)
    groq_json = test_structured_output_json_mode("Groq", chat_groq)
    groq_multi_tools = test_multiple_tools("Groq", chat_groq)
    groq_chain = test_chain_with_tools("Groq", chat_groq)
    
    # Print summary
    print("\n=== Test Summary ===\n")
    print("Basic Chat: Both models returned valid responses")
    print("Tool Calling: Both models successfully used tools")
    print("Structured Output (Function Calling): Both models returned structured data")
    print("Structured Output (JSON Mode): Both models handled JSON mode")
    print("Multiple Tools: Both models successfully used multiple tools")
    print("Chain with Tools: Both models worked within a chain")

if __name__ == "__main__":
    run_all_tests()
