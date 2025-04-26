"""
Comprehensive demonstration of all advanced features in the LangChain ASI integration.

This script showcases:
1. Basic chat functionality
2. Tool calling with different options
3. Structured output with function calling
4. Structured output with JSON mode
5. Complex schemas and nested structures
6. Streaming responses
7. Chains and advanced patterns
"""

import os
import json
from typing import List, Dict, Any, Optional, Type, Union, Callable, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import LangChain components
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Import the ChatASI model
from langchain_asi.chat_models import ChatASI

# Check if API key is available
asi_api_key = os.environ.get("ASI_API_KEY")
if not asi_api_key:
    raise ValueError("ASI_API_KEY environment variable is not set")

print("ASI API Key found. Initializing models...")

# Initialize the ChatASI model
chat_asi = ChatASI(
    model_name="asi1-mini",
    temperature=0,
    max_tokens=1000,
    verbose=True
)

# Initialize a streaming version of the model
streaming_chat_asi = ChatASI(
    model_name="asi1-mini",
    temperature=0,
    max_tokens=1000,
    streaming=True,
    verbose=True
)

print("Models initialized successfully.")

# Define example tools
@tool
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression.
    
    Args:
        expression: A mathematical expression (e.g., "2 + 2", "5 * 10")
        
    Returns:
        The result of the calculation
    """
    try:
        # Warning: eval can be dangerous in production code
        # This is just for demonstration purposes
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@tool
def search_database(query: str, database: str = "general") -> str:
    """Search a database for information.
    
    Args:
        query: The search query
        database: The database to search (general, scientific, historical)
        
    Returns:
        Search results
    """
    # Mock implementation
    databases = {
        "general": {
            "python": "Python is a high-level programming language known for its readability.",
            "langchain": "LangChain is a framework for developing applications powered by language models.",
            "ai": "Artificial Intelligence (AI) refers to systems that can perform tasks requiring human intelligence."
        },
        "scientific": {
            "quantum": "Quantum mechanics is a fundamental theory in physics that describes nature at atomic scales.",
            "relativity": "Einstein's theory of relativity describes the laws of physics in reference frames that are accelerating."
        },
        "historical": {
            "renaissance": "The Renaissance was a period of European cultural, artistic, political, and scientific rebirth.",
            "industrial revolution": "The Industrial Revolution was the transition to new manufacturing processes in the 18th century."
        }
    }
    
    if database not in databases:
        return f"Database '{database}' not found. Available databases: {', '.join(databases.keys())}"
    
    # Search for the query in the specified database
    db = databases[database]
    for key, value in db.items():
        if query.lower() in key.lower():
            return value
    
    return f"No results found for '{query}' in the {database} database."

# Define Pydantic models for structured output
class Author(BaseModel):
    """Information about an author."""
    name: str = Field(description="The author's full name")
    birth_year: Optional[int] = Field(None, description="The year the author was born")
    nationality: Optional[str] = Field(None, description="The author's nationality")

class BookReview(BaseModel):
    """A structured book review with nested author information."""
    title: str = Field(description="The title of the book being reviewed")
    author: Author = Field(description="Information about the book's author")
    publication_year: int = Field(description="Year the book was published")
    genre: str = Field(description="The book's genre (e.g., fiction, non-fiction, sci-fi)")
    rating: float = Field(description="Rating from 0.0 to 10.0")
    review: str = Field(description="Detailed review explaining the rating")
    recommended_for: List[str] = Field(description="Types of readers who would enjoy this book")

class MovieReview(BaseModel):
    """A structured movie review."""
    title: str = Field(description="The title of the movie being reviewed")
    rating: float = Field(description="Rating from 0.0 to 10.0")
    review: str = Field(description="Detailed review explaining the rating")
    recommended: bool = Field(description="Whether you would recommend this movie to others")

# Demo functions
def demo_basic_chat():
    """Demonstrate basic chat functionality."""
    print("\n" + "="*50)
    print("DEMO: Basic Chat")
    print("="*50)
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    
    response = chat_asi.invoke(messages)
    
    print(f"Response: {response.content}")
    return response

def demo_tool_calling_auto():
    """Demonstrate tool calling with auto tool choice."""
    print("\n" + "="*50)
    print("DEMO: Tool Calling (auto)")
    print("="*50)
    
    model_with_tools = chat_asi.bind_tools(
        [calculate, search_database],
        tool_choice="auto"
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant that can perform calculations and search databases."),
        HumanMessage(content="What is 25 * 16?")
    ]
    
    response = model_with_tools.invoke(messages)
    
    print(f"Response: {response.content}")
    print(f"Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'None'}")
    return response

def demo_tool_calling_specific():
    """Demonstrate tool calling with a specific tool."""
    print("\n" + "="*50)
    print("DEMO: Tool Calling (specific tool)")
    print("="*50)
    
    model_with_tools = chat_asi.bind_tools(
        [calculate, search_database],
        tool_choice="search_database"
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant that can perform calculations and search databases."),
        HumanMessage(content="Tell me about Python")
    ]
    
    response = model_with_tools.invoke(messages)
    
    print(f"Response: {response.content}")
    print(f"Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'None'}")
    return response

def demo_multiple_tools():
    """Demonstrate using multiple tools."""
    print("\n" + "="*50)
    print("DEMO: Multiple Tools")
    print("="*50)
    
    model_with_tools = chat_asi.bind_tools([calculate, search_database])
    
    messages = [
        SystemMessage(content="You are a helpful assistant that can perform calculations and search databases."),
        HumanMessage(content="Calculate 15 * 7 and also tell me about LangChain")
    ]
    
    response = model_with_tools.invoke(messages)
    
    print(f"Response: {response.content}")
    print(f"Tool calls: {response.tool_calls if hasattr(response, 'tool_calls') else 'None'}")
    return response

def demo_structured_output_function_calling():
    """Demonstrate structured output with function calling."""
    print("\n" + "="*50)
    print("DEMO: Structured Output (Function Calling)")
    print("="*50)
    
    structured_model = chat_asi.with_structured_output(
        MovieReview,
        method="function_calling"
    )
    
    messages = [
        SystemMessage(content="You are a movie critic who provides detailed reviews."),
        HumanMessage(content="Write a review for the movie 'The Matrix'")
    ]
    
    response = structured_model.invoke(messages)
    
    print(f"Response type: {type(response)}")
    print(f"Title: {response.title}")
    print(f"Rating: {response.rating}")
    print(f"Review: {response.review}")
    print(f"Recommended: {response.recommended}")
    return response

def demo_structured_output_json_mode():
    """Demonstrate structured output with JSON mode."""
    print("\n" + "="*50)
    print("DEMO: Structured Output (JSON Mode)")
    print("="*50)
    
    try:
        json_model = chat_asi.with_structured_output(
            MovieReview,
            method="json_mode"
        )
        
        messages = [
            SystemMessage(content="You are a movie critic who provides detailed reviews."),
            HumanMessage(content="Write a review for the movie 'Inception'. Return your response as a JSON object.")
        ]
        
        response = json_model.invoke(messages)
        
        print(f"Response type: {type(response)}")
        print(f"Title: {response.title}")
        print(f"Rating: {response.rating}")
        print(f"Review: {response.review}")
        print(f"Recommended: {response.recommended}")
        return response
    except Exception as e:
        print(f"Error with JSON mode: {e}")
        return None

def demo_complex_structured_output():
    """Demonstrate structured output with a complex nested schema."""
    print("\n" + "="*50)
    print("DEMO: Complex Structured Output")
    print("="*50)
    
    structured_model = chat_asi.with_structured_output(
        BookReview,
        method="function_calling"
    )
    
    messages = [
        SystemMessage(content="You are a literary critic who provides detailed book reviews."),
        HumanMessage(content="Write a review for the book '1984' by George Orwell")
    ]
    
    response = structured_model.invoke(messages)
    
    print(f"Response type: {type(response)}")
    print(f"Title: {response.title}")
    print(f"Author: {response.author.name}, {response.author.nationality}, born {response.author.birth_year}")
    print(f"Publication Year: {response.publication_year}")
    print(f"Genre: {response.genre}")
    print(f"Rating: {response.rating}")
    print(f"Review: {response.review[:100]}...")  # Truncated for readability
    print(f"Recommended for: {', '.join(response.recommended_for)}")
    return response

def demo_streaming():
    """Demonstrate streaming responses."""
    print("\n" + "="*50)
    print("DEMO: Streaming Responses")
    print("="*50)
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Write a short poem about artificial intelligence.")
    ]
    
    print("Streaming response:")
    for chunk in streaming_chat_asi.stream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")

def demo_chain_with_tools():
    """Demonstrate using tools within a chain."""
    print("\n" + "="*50)
    print("DEMO: Chain with Tools")
    print("="*50)
    
    model_with_tools = chat_asi.bind_tools([calculate, search_database])
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can perform calculations and search databases."),
        ("human", "{input}")
    ])
    
    # Create a chain
    chain = prompt | model_with_tools | StrOutputParser()
    
    # Run the chain
    response = chain.invoke({"input": "Calculate 42 * 18 and tell me about artificial intelligence"})
    
    print(f"Chain Response: {response}")
    return response

def run_all_demos():
    """Run all demonstration functions."""
    print("\nRunning comprehensive feature demonstrations for LangChain ASI integration...")
    
    # Run all demos
    demo_basic_chat()
    demo_tool_calling_auto()
    demo_tool_calling_specific()
    demo_multiple_tools()
    demo_structured_output_function_calling()
    demo_structured_output_json_mode()
    demo_complex_structured_output()
    demo_streaming()
    demo_chain_with_tools()
    
    print("\n" + "="*50)
    print("All demonstrations completed successfully!")
    print("="*50)
    print("\nThe LangChain ASI integration now has full feature parity with other LangChain integrations.")
    print("It supports all advanced features including tool calling, structured output, and streaming.")

if __name__ == "__main__":
    run_all_demos()
