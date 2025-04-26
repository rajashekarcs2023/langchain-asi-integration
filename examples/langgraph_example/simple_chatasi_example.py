"""Simple example of using ChatASI without LangGraph."""
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_asi import ChatASI

# Set your API key - for a real implementation, use environment variables
# The ASI API key is loaded from the ASI_API_KEY environment variable
# You can set this with: export ASI_API_KEY="your-api-key"
# The ASI API key should be set in your environment variables as ASI_API_KEY
asi_api_key = os.environ.get("ASI_API_KEY")
if not asi_api_key:
    raise ValueError("ASI_API_KEY environment variable is not set")

# Initialize the ChatASI model
chat = ChatASI(model_name="asi1-mini")  # Choose the appropriate model

# Define a simple function to process a query
def process_query(query: str) -> str:
    """Process a user query using ChatASI."""
    # Create the messages
    messages = [
        SystemMessage(content="You are a helpful assistant that provides accurate and concise information."),
        HumanMessage(content=query)
    ]
    
    # Invoke the chat model
    try:
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    # Example question
    question = "What are the financial prospects for Tesla in the next year?"
    
    # Process the query
    response = process_query(question)
    
    # Print the response
    print(f"\nQuestion: {question}\n")
    print(f"Response: {response}")
