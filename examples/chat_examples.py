"""Examples of using the ChatASI class with different configurations."""

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_asi.langchain_asi.chat_models import ChatASI

# Load environment variables from .env file
load_dotenv()


def basic_chat_example():
    """Basic example of using ChatASI."""
    # Initialize the chat model
    # Note: This will automatically use ASI_API_KEY from environment variables
    # and select the appropriate API endpoint based on the model name
    chat = ChatASI(model_name="asi1-mini")

    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ]

    # Generate a response
    response = chat.invoke(messages)
    print(f"Response: {response.content}\n")

    # Check token usage
    if response.usage_metadata:
        print(f"Token usage: {response.usage_metadata}\n")


def streaming_example():
    """Example of using streaming with ChatASI."""
    # Initialize the chat model with streaming enabled
    chat = ChatASI(model_name="asi1-mini", streaming=True)

    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Write a short poem about artificial intelligence."),
    ]

    # Stream the response
    print("Streaming response:")
    for chunk in chat.stream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")


def tool_calling_example():
    """Example of using tool calling with ChatASI."""
    # Define a simple weather tool
    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get the weather for a location.

        Args:
            location: The city and state, e.g. San Francisco, CA
            unit: The unit of temperature, either celsius or fahrenheit

        Returns:
            The weather forecast for the specified location
        """
        # In a real application, this would call a weather API
        return f"The weather in {location} is 22°{unit[0].upper()}, sunny with a slight breeze."

    # Define a simple calculator tool
    def calculator(expression: str) -> str:
        """Calculate the result of a mathematical expression.

        Args:
            expression: A mathematical expression as a string, e.g. "2 + 2"

        Returns:
            The result of the calculation
        """
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {str(e)}"

    # Initialize the chat model and bind tools
    chat = ChatASI(model_name="asi1-mini")
    chat_with_tools = chat.bind_tools([get_weather, calculator])

    # Create messages
    messages = [
        SystemMessage(
            content="You are a helpful assistant with access to tools. Use them when appropriate."
        ),
        HumanMessage(
            content="What's the weather in New York? Also, what's 15 * 24?"
        ),
    ]

    # Generate a response with tool calls
    response = chat_with_tools.invoke(messages)
    print(f"Response with tool calls: {response}\n")


def langchain_chain_example():
    """Example of using ChatASI in a LangChain chain."""
    # Initialize the chat model
    chat = ChatASI(model_name="asi1-mini")

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that summarizes text."),
            ("user", "Summarize the following text in {word_count} words:\n{text}"),
        ]
    )

    # Create a chain
    chain = prompt | chat | StrOutputParser()

    # Run the chain
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". 
    This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    """

    result = chain.invoke({"text": text, "word_count": "30"})
    print(f"Summarization result: {result}\n")


def async_example():
    """Example of using ChatASI asynchronously."""
    import asyncio

    async def generate_async():
        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What are three benefits of artificial intelligence?"),
        ]

        # Generate a response asynchronously
        response = await chat.ainvoke(messages)
        print(f"Async response: {response.content}\n")

        # Stream a response asynchronously
        print("Async streaming response:")
        async for chunk in chat.astream(messages):
            print(chunk.content, end="", flush=True)
        print("\n")

    # Run the async example
    asyncio.run(generate_async())


def custom_parameters_example():
    """Example of using ChatASI with custom parameters."""
    # Initialize the chat model with custom parameters
    chat = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=100,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        timeout=30,
        verbose=True,
    )

    # Create messages
    messages = [
        SystemMessage(content="You are a creative assistant."),
        HumanMessage(content="Generate a creative name for a tech startup."),
    ]

    # Generate a response
    response = chat.invoke(messages)
    print(f"Response with custom parameters: {response.content}\n")


if __name__ == "__main__":
    print("\n=== Basic Chat Example ===")
    basic_chat_example()

    print("\n=== Streaming Example ===")
    streaming_example()

    print("\n=== Tool Calling Example ===")
    tool_calling_example()

    print("\n=== LangChain Chain Example ===")
    langchain_chain_example()

    print("\n=== Async Example ===")
    async_example()

    print("\n=== Custom Parameters Example ===")
    custom_parameters_example()
