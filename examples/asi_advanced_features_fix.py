"""
ASI Advanced Features Fix

This script provides fixes for the advanced features in the ChatASI class
to ensure they work properly with LangChain's abstractions.
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Union, Sequence, Type, Literal
from pydantic import BaseModel, Field
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import Runnable, RunnableConfig, RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.language_models import LanguageModelInput
from langchain_core.utils.pydantic import is_basemodel_subclass
from operator import itemgetter

# Load environment variables
load_dotenv()

# Define a simple Pydantic model for structured output
class WeatherInfo(BaseModel):
    """Information about the weather in a location."""
    
    location: str = Field(description="The location for the weather information")
    temperature: float = Field(description="The temperature in Celsius")
    conditions: str = Field(description="The weather conditions (e.g., sunny, rainy)")
    humidity: Optional[float] = Field(None, description="The humidity percentage")

# Define a simple tool
class GetWeather(BaseModel):
    """Get the current weather in a given location."""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

def fix_bind_tools_method(chat_asi: ChatASI):
    """Fix the bind_tools method in the ChatASI class."""
    
    # Define a new bind_tools method that works correctly
    def new_bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        # Convert tools to the format expected by ASI API
        asi_tools = []
        for tool in formatted_tools:
            if "function" in tool:
                function_def = tool["function"]
                asi_tools.append({
                    "type": "function",
                    "function": function_def
                })
        
        # Handle tool_choice parameter
        asi_tool_choice = None
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                asi_tool_choice = "auto"
            elif isinstance(tool_choice, str) and tool_choice not in ("auto", "none"):
                # If tool_choice is a string (tool name), format it properly
                asi_tool_choice = {
                    "type": "function", 
                    "function": {"name": tool_choice}
                }
            elif isinstance(tool_choice, bool) and tool_choice:
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                # If tool_choice is True and there's only one tool, use that tool
                tool_name = formatted_tools[0]["function"]["name"]
                asi_tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }
            else:
                # Pass through other values (like dicts or None)
                asi_tool_choice = tool_choice
        
        # Add tools and tool_choice to kwargs
        if asi_tools:
            kwargs["tools"] = asi_tools
        if asi_tool_choice is not None:
            kwargs["tool_choice"] = asi_tool_choice
        
        # Use the bind method from BaseChatModel
        return self.bind(**kwargs)
    
    # Replace the bind_tools method in the ChatASI class
    chat_asi.bind_tools = new_bind_tools.__get__(chat_asi, ChatASI)
    
    return chat_asi

def fix_with_structured_output_method(chat_asi: ChatASI):
    """Fix the with_structured_output method in the ChatASI class."""
    
    # Define a new with_structured_output method that works correctly
    def new_with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Optional[str] = None,
        strict: Optional[bool] = None,
        include_raw: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:
                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a Pydantic model,
                - a Python dictionary,
                - a string describing the schema,
                - a string describing the schema in JSON format.
            method: The method to use for structured output. If not provided, will
                automatically select an appropriate method based on the model and schema.
                Valid options are:
                - "function_calling": Use function calling
                - "json_mode": Use JSON mode
                - "tool_calling": Use tool calling
                - "jsonformer": Use jsonformer
            strict: Whether to enforce the schema strictly. If True, will raise an error
                if the output does not match the schema. If False, will attempt to
                coerce the output to match the schema. If not provided, will default to
                False.
            include_raw: Whether to include the raw model output in the output.
            name: The name of the function to use for function calling. If not provided,
                will use the name of the Pydantic model if provided, or a default name.
            **kwargs: Additional parameters to pass to the model.

        Returns:
            A runnable that returns outputs formatted to match the given schema.

        Examples:
            .. code-block:: python

                from langchain_core.pydantic_v1 import BaseModel, Field
                from langchain_asi import ChatASI

                # Defining our response schema
                class Album(BaseModel):
                    name: str = Field(description="Name of the album")
                    artist: str = Field(description="Name of the artist")
                    year: int = Field(description="Year the album was released")
                    genre: str = Field(description="Genre of the album")

                # Creating our model
                llm = ChatASI(model_name="asi1-mini")

                # Creating a structured output model
                structured_llm = llm.with_structured_output(Album)

                # We can now use the model to get structured outputs
                structured_llm.invoke(
                    "What album is known as the best selling album of all time?"
                )
                # -> Album(name='Thriller', artist='Michael Jackson', year=1982, genre='Pop')

                # We can also get the raw model output alongside the structured output
                structured_llm = llm.with_structured_output(Album, include_raw=True)

                structured_llm.invoke(
                    "What album is known as the best selling album of all time?"
                )
                # -> {
                #     'raw': AIMessage(content=''),
                #     'parsed': Album(name='Thriller', artist='Michael Jackson', year=1982, genre='Pop'),
                #     'parsing_error': None
                # }
        """
        if schema is None:
            raise ValueError("schema must be provided")

        # If schema is a Pydantic model, convert it to a tool
        if is_basemodel_subclass(schema):
            pydantic_schema = schema
            name = name or pydantic_schema.__name__
            output_parser = PydanticToolsParser(
                first_tool_only=True, tools=[pydantic_schema]
            )
            
            # Convert the Pydantic model to a tool
            tool = convert_to_openai_tool(pydantic_schema)
            
            # Create a bound model with the tool
            llm = self.bind(
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": name}},
                ls_structured_output_format={
                    "schema": tool,
                    "kwargs": {"method": "function_calling"},
                },
            )
        else:
            # Handle other schema types
            output_parser = PydanticOutputParser(pydantic_schema=schema)
            
            # Create a bound model with JSON mode
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "schema": schema,
                    "kwargs": {"method": "json_mode"},
                },
            )
        
        # Handle include_raw parameter
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser
    
    # Replace the with_structured_output method in the ChatASI class
    chat_asi.with_structured_output = new_with_structured_output.__get__(chat_asi, ChatASI)
    
    return chat_asi

def fix_ainvoke_method(chat_asi: ChatASI):
    """Fix the ainvoke method in the ChatASI class."""
    
    # Define a new ainvoke method that works correctly
    async def new_ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Asynchronously invoke the chat model.

        Args:
            input: The input messages to send to the ASI API.
            config: A RunnableConfig to use for the invocation.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            The generated message.
        """
        messages = self._convert_input_to_messages(input)
        generation = await self.agenerate(
            messages=messages,
            callbacks=config.get("callbacks") if config else None,
            **kwargs,
        )
        return generation.generations[0].message
    
    # Replace the ainvoke method in the ChatASI class
    chat_asi.ainvoke = new_ainvoke.__get__(chat_asi, ChatASI)
    
    return chat_asi

async def test_fixed_asi_advanced_features():
    """Test the fixed advanced features in the ChatASI class."""
    print("=" * 80)
    print("FIXED ASI ADVANCED FEATURES TEST")
    print("=" * 80)
    
    # Initialize the ChatASI model
    asi_chat = ChatASI(model_name="asi1-mini", verbose=True)
    
    # Fix the advanced features methods
    asi_chat = fix_bind_tools_method(asi_chat)
    asi_chat = fix_with_structured_output_method(asi_chat)
    asi_chat = fix_ainvoke_method(asi_chat)
    
    # Test 1: bind_tools
    print("\n[Test 1: bind_tools]")
    try:
        # Bind tools to the model
        model_with_tools = asi_chat.bind_tools([GetWeather])
        
        # Print the model configuration
        print(f"Model with tools: {model_with_tools}")
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant that can provide weather information."),
            HumanMessage(content="What's the weather like in San Francisco?")
        ]
        
        # Use the model with tools
        response = await model_with_tools.ainvoke(messages)
        print(f"Response: {response}")
        
        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            print("\nTool calls:")
            for tool_call in response.tool_calls:
                print(f"  Tool: {tool_call.get('name')}")
                print(f"  Arguments: {tool_call.get('args')}")
        elif hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
            print("\nTool calls (from additional_kwargs):")
            for tool_call in response.additional_kwargs["tool_calls"]:
                print(f"  Tool: {tool_call.get('function', {}).get('name')}")
                print(f"  Arguments: {tool_call.get('function', {}).get('arguments')}")
        else:
            print("\nNo tool calls were made.")
            print(f"Response content: {response.content}")
    except Exception as e:
        print(f"Error in bind_tools test: {e}")
    
    # Test 2: with_structured_output
    print("\n[Test 2: with_structured_output]")
    try:
        # Create a model with structured output
        structured_model = asi_chat.with_structured_output(WeatherInfo)
        
        # Print the model configuration
        print(f"Structured model: {structured_model}")
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant that can provide weather information."),
            HumanMessage(content="What's the weather like in Paris?")
        ]
        
        # Use the model with structured output
        response = await structured_model.ainvoke(messages)
        print(f"Structured response: {response}")
        
        # Test with include_raw=True
        structured_model_with_raw = asi_chat.with_structured_output(WeatherInfo, include_raw=True)
        response_with_raw = await structured_model_with_raw.ainvoke(messages)
        print("\nStructured response with raw:")
        print(f"  Raw: {response_with_raw.get('raw')}")
        print(f"  Parsed: {response_with_raw.get('parsed')}")
        print(f"  Parsing error: {response_with_raw.get('parsing_error')}")
    except Exception as e:
        print(f"Error in with_structured_output test: {e}")
    
    # Test 3: ainvoke with different input formats
    print("\n[Test 3: ainvoke with different input formats]")
    try:
        # Test with a list of messages (standard format)
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?")
        ]
        response = await asi_chat.ainvoke(messages)
        print(f"List of messages response: {response.content}")
        
        # Test with a single message
        message = HumanMessage(content="What is the capital of Italy?")
        response = await asi_chat.ainvoke(message)
        print(f"Single message response: {response.content}")
        
        # Test with a string
        try:
            response = await asi_chat.ainvoke("What is the capital of Germany?")
            print(f"String response: {response.content}")
        except Exception as e:
            print(f"String input error: {e}")
    except Exception as e:
        print(f"Error in ainvoke test: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_fixed_asi_advanced_features())
