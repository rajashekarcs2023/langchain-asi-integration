"""Unit tests for the OpenAI adapter."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langchain_asi.chat_models import ChatASI
from langchain_asi.openai_adapter import OpenAICompatibleASI, create_openai_compatible_model


class TestOpenAIAdapter:
    """Test the OpenAI adapter."""

    def setup_method(self):
        """Set up the test environment."""
        # Create a patcher for the environment validation
        self.env_patcher = patch.dict(os.environ, {"ASI_API_KEY": "fake-api-key"})
        self.env_patcher.start()
        
        # Create a mock ChatASI model
        self.mock_model = MagicMock(spec=ChatASI)
        
        # Mock the invoke method to return an AIMessage
        self.mock_model.invoke.return_value = AIMessage(content="This is a test response.")
        
        # Create the adapter
        self.adapter = OpenAICompatibleASI(self.mock_model)
        
    def teardown_method(self):
        """Clean up after the test."""
        self.env_patcher.stop()

    @patch("langchain_asi.openai_adapter.OpenAICompatibleASI._parse_with_fallback")
    def test_adapter_initialization(self, mock_parse):
        """Test that the adapter initializes correctly."""
        # Check that the model was set correctly
        assert self.adapter.model == self.mock_model
        
        # Check that the tools list is empty
        assert self.adapter.tools == []
        
        # Check that the tool_choice is None
        assert self.adapter._tool_choice is None

    def test_adapter_invoke(self):
        """Test that the adapter invokes the model correctly."""
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, how are you?"),
        ]
        
        # Invoke the adapter
        result = self.adapter.invoke(messages)
        
        # Check that the model was invoked
        self.mock_model.invoke.assert_called_once_with(messages)
        
        # Check that the result is correct
        assert result.content == "This is a test response."

    def test_bind_tools(self):
        """Test binding tools to the adapter."""
        # Define a tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        
        # Bind the tools to the adapter
        result = self.adapter.bind_tools(tools)
        
        # Check that the tools were set correctly
        assert result.tools == tools
        
        # Check that the tool_choice is None
        assert result._tool_choice is None

    def test_bind_tools_with_tool_choice(self):
        """Test binding tools with a specific tool choice."""
        # Define tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        
        # Bind the tools with a specific tool choice
        result = self.adapter.bind_tools(tools, tool_choice="get_weather")
        
        # Check that the tools were set correctly
        assert result.tools == tools
        
        # Check that the tool_choice is correct
        assert result._tool_choice == "get_weather"

    def test_bind_functions(self):
        """Test binding functions to the adapter."""
        # Define functions
        functions = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]
        
        # Bind the functions to the adapter
        result = self.adapter.bind_functions(functions)
        
        # Check that the tools were set correctly
        assert len(result.tools) == 1
        assert result.tools[0]["type"] == "function"
        assert result.tools[0]["function"]["name"] == "get_weather"

    def test_bind_functions_with_function_call(self):
        """Test binding functions with a specific function call."""
        # Define functions
        functions = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]
        
        # Bind the functions with a specific function call
        result = self.adapter.bind_functions(functions, function_call="get_weather")
        
        # Check that the tools were set correctly
        assert len(result.tools) == 1
        
        # Check that the tool_choice is correct
        assert result._tool_choice["type"] == "function"
        assert result._tool_choice["function"]["name"] == "get_weather"

    def test_with_structured_output(self):
        """Test structured output with the adapter."""
        # Define a schema
        class MovieReview(BaseModel):
            title: str = Field(description="The title of the movie")
            rating: float = Field(description="Rating from 0.0 to 10.0")
            review: str = Field(description="Detailed review explaining the rating")

        # Create a model with structured output
        result = self.adapter.with_structured_output(MovieReview)
        
        # Check that the schema was set correctly
        assert result._structured_output_schema == MovieReview

    @patch("langchain_asi.openai_adapter.OpenAICompatibleASI._parse_with_fallback")
    def test_parse_with_fallback_json(self, mock_parse):
        """Test parsing JSON with fallback methods."""
        # Define a schema
        class TestSchema(BaseModel):
            name: str
            age: int

        # Set up the mock to return a parsed object
        mock_parse.return_value = TestSchema(name="Test", age=30)
        
        # Parse a JSON string
        json_str = '{"name": "Test", "age": 30}'
        result = self.adapter._parse_with_fallback(json_str, TestSchema)
        
        # Check that the result is correct
        assert result.name == "Test"
        assert result.age == 30

    @patch("langchain_asi.openai_adapter.OpenAICompatibleASI._parse_with_fallback")
    def test_parse_with_fallback_code_block(self, mock_parse):
        """Test parsing JSON from a code block with fallback methods."""
        # Define a schema
        class TestSchema(BaseModel):
            name: str
            age: int

        # Set up the mock to return a parsed object
        mock_parse.return_value = TestSchema(name="Test", age=30)
        
        # Parse a JSON string in a code block
        json_str = """```json
        {"name": "Test", "age": 30}
        ```"""
        result = self.adapter._parse_with_fallback(json_str, TestSchema)
        
        # Check that the result is correct
        assert result.name == "Test"
        assert result.age == 30

    def test_create_supervisor(self):
        """Test creating a supervisor function."""
        # Define a schema for the routing decision
        class RouteSchema(BaseModel):
            next: str = Field(description="The next agent to call")
            reasoning: str = Field(description="Explanation for why this agent should act next")

        # Create a supervisor function
        supervisor = self.adapter.create_supervisor(
            system_prompt="You are a supervisor.",
            members=["Agent1", "Agent2"],
            schema=RouteSchema
        )
        
        # Check that the supervisor function was created
        assert callable(supervisor)

    def test_create_openai_compatible_model(self):
        """Test creating an OpenAI-compatible model."""
        # Create a mock ChatASI model
        mock_model = MagicMock(spec=ChatASI)
        
        # Create an OpenAI-compatible model
        adapter = create_openai_compatible_model(mock_model)
        
        # Check that the adapter was created correctly
        assert isinstance(adapter, OpenAICompatibleASI)
        assert adapter.model == mock_model
