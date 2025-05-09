"""Basic usage examples for langchain-asi."""
import os
from dotenv import load_dotenv
from langchain_asi import ChatASI, ASIJsonOutputParserWithValidation
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional, Type

# Load environment variables
load_dotenv()

# Check if ASI_API_KEY is set
if not os.environ.get("ASI_API_KEY"):
    print("Error: ASI_API_KEY environment variable not found.")
    print("Please create a .env file with your ASI_API_KEY or set it directly in your environment.")
    print("Example .env file content: ASI_API_KEY=your-api-key-here")
    exit(1)

# Set your API key - for a real implementation, use environment variables
# os.environ["ASI_API_KEY"] = "your-api-key"  # Replace with your actual API key


def basic_example():
    """Demonstrate basic usage of ChatASI."""
    # Initialize the chat model
    chat = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
    )
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Tell me a short joke about programming.")
    ]
    
    # Generate response
    response = chat.invoke(messages)
    print("Basic example response:")
    print(response.content)
    print()


def tool_calling_example():
    """Demonstrate tool calling with ChatASI."""
    # Define a weather tool
    class GetWeather(BaseModel):
        """Get the current weather in a given location"""
        
        location: str = Field(description="The city and state, e.g. San Francisco, CA")
        unit: Optional[str] = Field(
            default="fahrenheit", 
            description="The unit of temperature, either 'celsius' or 'fahrenheit'"
        )
    
    # Initialize the chat model
    chat = ChatASI(
        model_name="asi1-mini",
        temperature=0.1,
    )
    
    # Bind the tool to the model
    chat_with_tools = chat.bind_tools([GetWeather])
    
    # Invoke the model with a question that requires the tool
    response = chat_with_tools.invoke(
        "What's the weather like in Seattle?"
    )
    
    print("Tool calling example response:")
    print(f"Content: {response.content}")
    print(f"Tool calls: {response.tool_calls}")
    print()


def structured_output_example():
    """Demonstrate structured output with ChatASI."""
    # Define a structured output schema
    class MovieReview(BaseModel):
        """Movie review with title, year, and review text."""
        
        title: str = Field(description="The title of the movie")
        year: int = Field(description="The year the movie was released")
        genre: List[str] = Field(description="The genres of the movie")
        review: str = Field(description="A brief review of the movie")
        rating: int = Field(description="Rating from 1-10, with 10 being the best")
    
    # Initialize the chat model
    chat = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
    )
    
    print("Structured output example:")
    
    # Try the tool-based approach first (recommended)
    try:
        # Bind the MovieReview schema as a tool
        chat_with_tools = chat.bind_tools([MovieReview])
        
        # Create a prompt that instructs the model to use the tool
        tool_prompt = """Write a review for The Matrix. Use the MovieReview tool to structure your response."""
        
        # Invoke the model with the tool
        tool_response = chat_with_tools.invoke(tool_prompt)
        
        # Extract the tool calls from the response
        if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
            # Get the first tool call
            tool_call = tool_response.tool_calls[0]
            review_data = tool_call.get('args', {})
            
            # Create a MovieReview object
            review = MovieReview(
                title=review_data.get('title', 'Unknown'),
                year=int(review_data.get('year', 0)),
                genre=review_data.get('genre', []),
                review=review_data.get('review', ''),
                rating=int(review_data.get('rating', 0))
            )
            
            print("\nUsing tool calling (recommended approach):")
            print(f"Title: {review.title}")
            print(f"Year: {review.year}")
            print(f"Genres: {', '.join(review.genre)}")
            print(f"Rating: {review.rating}/10")
            print(f"Review: {review.review}")
        else:
            print("Tool calling approach failed: No tool calls in response")
            raise ValueError("No tool calls in response")
    except Exception as e:
        print(f"Tool calling approach failed: {e}")
        
        # Fall back to the JSON approach
        print("\nFalling back to JSON parsing approach:")
        
        # Create a structured output chain with more explicit instructions
        prompt = """Write a review for The Matrix. 
        
        Your response must be a valid JSON object with the following fields:
        - title: The title of the movie
        - year: The year the movie was released (as an integer)
        - genre: A list of genres for the movie
        - review: A brief review of the movie
        - rating: A rating from 1-10, with 10 being the best (as an integer)
        
        Format your entire response as a JSON object with no additional text.
        Wrap your response in triple backticks with the json tag like this:
        ```json
        {"your": "json here"}
        ```
        """
        
        # First try using with_structured_output method directly
        try:
            structured_model = chat.with_structured_output(MovieReview)
            review = structured_model.invoke(prompt)
            
            print(f"Title: {review.title}")
            print(f"Year: {review.year}")
            print(f"Genres: {', '.join(review.genre)}")
            print(f"Rating: {review.rating}/10")
            print(f"Review: {review.review}")
        except Exception as e:
            print(f"First attempt failed: {e}")
            
            # If that fails, try using a custom parser
            try:
                print("Trying with custom parser...")
                # Use the ASIJsonOutputParser directly
                class MovieReviewOutputParser(ASIJsonOutputParserWithValidation):
                    pydantic_object: Type[BaseModel] = Field(default=MovieReview)
                
                parser = MovieReviewOutputParser()
                
                # Create a chain with the parser
                prompt_template = ChatPromptTemplate.from_template("{input}")
                chain = prompt_template | chat | parser
                
                # Generate a structured movie review
                review = chain.invoke({"input": prompt})
                
                print(f"Title: {review.title}")
                print(f"Year: {review.year}")
                print(f"Genres: {', '.join(review.genre)}")
                print(f"Rating: {review.rating}/10")
                print(f"Review: {review.review}")
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                
                # Last resort: try to parse manually
                try:
                    print("Attempting to parse manually...")
                    # Get raw response
                    raw_response = chat.invoke(prompt)
                    
                    # Try to extract JSON from the response
                    import json
                    import re
                    
                    text = raw_response.content
                    
                    # Try to extract JSON from the text using regex
                    json_match = re.search(r"```json\s*(.+?)\s*```", text, re.DOTALL)
                    if json_match:
                        text = json_match.group(1).strip()
                    else:
                        # Look for JSON objects within the text
                        json_match = re.search(r"\{.+\}", text, re.DOTALL)
                        if json_match:
                            text = json_match.group(0).strip()
                    
                    # Parse the JSON
                    json_object = json.loads(text)
                    
                    # Create a MovieReview object
                    if isinstance(json_object, dict):
                        # Handle case where 'genre' might be a string instead of a list
                        genre = json_object.get("genre", [])
                        if isinstance(genre, str):
                            # Split by commas or other common separators
                            genre = [g.strip() for g in re.split(r"[,|;]", genre)]
                        
                        review = MovieReview(
                            title=json_object.get("title", "Unknown"),
                            year=int(json_object.get("year", 0)),
                            genre=genre,
                            review=json_object.get("review", ""),
                            rating=int(json_object.get("rating", 0))
                        )
                        
                        print(f"Title: {review.title}")
                        print(f"Year: {review.year}")
                        print(f"Genres: {', '.join(review.genre)}")
                        print(f"Rating: {review.rating}/10")
                        print(f"Review: {review.review}")
                    else:
                        print(f"Failed to parse JSON: not a dictionary")
                except Exception as e3:
                    print(f"All parsing attempts failed: {e3}")
    print()


def streaming_example():
    """Demonstrate streaming with ChatASI."""
    chat = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        streaming=True,
    )
    
    messages = [
        HumanMessage(content="Explain quantum computing in simple terms.")
    ]
    
    print("Streaming example response:")
    for chunk in chat.stream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print("Running langchain-asi examples...\n")
    
    try:
        basic_example()
        tool_calling_example()
        structured_output_example()
        streaming_example()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your ASI_API_KEY environment variable correctly.")