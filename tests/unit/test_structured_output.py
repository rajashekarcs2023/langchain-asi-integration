"""Unit tests for structured output functionality."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langchain_asi.chat_models import ChatASI


class SimpleSchema(BaseModel):
    """A simple schema for testing."""
    name: str = Field(description="The name")
    age: int = Field(description="The age")


class NestedSchema(BaseModel):
    """A schema with nested fields for testing."""
    title: str = Field(description="The title")
    author: SimpleSchema = Field(description="The author information")
    year: int = Field(description="The publication year")


class ComplexSchema(BaseModel):
    """A complex schema with lists and optional fields for testing."""
    name: str = Field(description="The name")
    tags: list[str] = Field(description="List of tags")
    metadata: dict = Field(description="Metadata dictionary")
    optional_field: str = Field(default=None, description="An optional field")


class TestStructuredOutput:
    """Test the structured output functionality."""

    def setup_method(self):
        """Set up the test environment."""
        # Create a patcher for the environment validation
        self.env_patcher = patch.dict(os.environ, {"ASI_API_KEY": "fake-api-key"})
        self.env_patcher.start()
        
        # Create the chat model with the mocked environment
        self.chat = ChatASI(model_name="asi1-mini")
        
        # Mock the HTTP clients
        self.chat.client = MagicMock()
        self.chat.async_client = MagicMock()
        
    def teardown_method(self):
        """Clean up after the test."""
        self.env_patcher.stop()

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_with_structured_output_function_calling(self, mock_invoke):
        """Test with_structured_output using function calling.
        
        Note: For ASI models, function_calling uses JSON mode with appropriate instructions.
        """
        # Mock the JSON response that would be returned by the model
        mock_response = AIMessage(
            content='{"name": "John Doe", "age": 30}'
        )
        mock_invoke.return_value = mock_response

        # Create a model with structured output
        structured_model = self.chat.with_structured_output(
            SimpleSchema,
            method="function_calling"
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about John Doe")
        ]
        
        result = structured_model.invoke(messages)
        
        # Check that the result is a SimpleSchema
        assert isinstance(result, SimpleSchema)
        assert result.name == "John Doe"
        assert result.age == 30

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_with_structured_output_json_mode(self, mock_invoke):
        """Test with_structured_output using JSON mode."""
        # Mock the response
        mock_response = AIMessage(
            content="""```json
{
  "name": "John Doe",
  "age": 30
}
```"""
        )
        mock_invoke.return_value = mock_response

        # Create a model with structured output
        structured_model = self.chat.with_structured_output(
            SimpleSchema,
            method="json_mode"
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about John Doe")
        ]
        
        result = structured_model.invoke(messages)
        
        # Check that the result is a SimpleSchema
        assert isinstance(result, SimpleSchema)
        assert result.name == "John Doe"
        assert result.age == 30

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_with_structured_output_nested_schema(self, mock_invoke):
        """Test with_structured_output using a nested schema.
        
        Note: For ASI models, we use JSON mode with appropriate instructions.
        """
        # Mock the JSON response that would be returned by the model
        mock_response = AIMessage(
            content='''
            {
                "title": "The Great Gatsby",
                "author": {
                    "name": "F. Scott Fitzgerald",
                    "age": 45
                },
                "year": 1925
            }
            '''
        )
        mock_invoke.return_value = mock_response

        # Create a model with structured output
        structured_model = self.chat.with_structured_output(
            NestedSchema,
            method="function_calling"
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about The Great Gatsby")
        ]
        
        result = structured_model.invoke(messages)
        
        # Check that the result is a NestedSchema
        assert isinstance(result, NestedSchema)
        assert result.title == "The Great Gatsby"
        assert isinstance(result.author, SimpleSchema)
        assert result.author.name == "F. Scott Fitzgerald"
        assert result.author.age == 45
        assert result.year == 1925

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_with_structured_output_complex_schema(self, mock_invoke):
        """Test with_structured_output using a complex schema.
        
        Note: For ASI models, we use JSON mode with appropriate instructions.
        """
        # Mock the JSON response that would be returned by the model
        mock_response = AIMessage(
            content='''
            {
                "name": "Project X",
                "tags": ["important", "urgent"],
                "metadata": {
                    "description": "A complex project",
                    "status": "active",
                    "team_members": [
                        {
                            "name": "Alice",
                            "role": "Developer"
                        },
                        {
                            "name": "Bob",
                            "role": "Designer"
                        }
                    ]
                },
                "optional_field": "This is optional"
            }
            '''
        )
        mock_invoke.return_value = mock_response

        # Create a model with structured output
        structured_model = self.chat.with_structured_output(
            ComplexSchema,
            method="function_calling"
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about Project X")
        ]
        
        result = structured_model.invoke(messages)
        
        # Check that the result is a ComplexSchema
        assert isinstance(result, ComplexSchema)
        assert result.name == "Project X"
        assert isinstance(result.tags, list)
        assert "important" in result.tags
        assert isinstance(result.metadata, dict)
        assert result.metadata["description"] == "A complex project"
        assert result.metadata["status"] == "active"
        assert "team_members" in result.metadata
        assert result.optional_field == "This is optional"

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_with_structured_output_missing_optional_field(self, mock_invoke):
        """Test with_structured_output with missing optional fields.
        
        Note: For ASI models, we use JSON mode with appropriate instructions.
        """
        # Mock the JSON response that would be returned by the model
        mock_response = AIMessage(
            content='''
            {
                "name": "Complex Object",
                "tags": ["tag1", "tag2"],
                "metadata": {
                    "key1": "value1"
                }
            }
            '''
        )
        mock_invoke.return_value = mock_response

        # Create a model with structured output
        structured_model = self.chat.with_structured_output(
            ComplexSchema,
            method="function_calling"
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Give me a complex object with missing fields")
        ]
        
        result = structured_model.invoke(messages)
        
        # Check that the result is a ComplexSchema
        assert isinstance(result, ComplexSchema)
        assert result.name == "Complex Object"
        assert len(result.tags) == 2
        assert isinstance(result.metadata, dict)
        assert result.metadata["key1"] == "value1"
        assert result.optional_field is None  # Default value for optional field

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_with_structured_output_json_mode_no_code_block(self, mock_invoke):
        """Test with_structured_output using JSON mode without code blocks."""
        # Mock the response without code blocks
        mock_response = AIMessage(
            content="""
{
  "name": "John Doe",
  "age": 30
}
"""
        )
        mock_invoke.return_value = mock_response

        # Create a model with structured output
        structured_model = self.chat.with_structured_output(
            SimpleSchema,
            method="json_mode"
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about John Doe")
        ]
        
        result = structured_model.invoke(messages)
        
        # Check that the result is a SimpleSchema
        assert isinstance(result, SimpleSchema)
        assert result.name == "John Doe"
        assert result.age == 30

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_with_structured_output_json_mode_with_text(self, mock_invoke):
        """Test with_structured_output using JSON mode with surrounding text."""
        # Mock the response with text around the JSON
        mock_response = AIMessage(
            content="""
Here's the information about John Doe:

```json
{
  "name": "John Doe",
  "age": 30
}
```

Let me know if you need anything else!
"""
        )
        mock_invoke.return_value = mock_response

        # Create a model with structured output
        structured_model = self.chat.with_structured_output(
            SimpleSchema,
            method="json_mode"
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about John Doe")
        ]
        
        result = structured_model.invoke(messages)
        
        # Check that the result is a SimpleSchema
        assert isinstance(result, SimpleSchema)
        assert result.name == "John Doe"
        assert result.age == 30

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_with_structured_output_invalid_json(self, mock_invoke):
        """Test with_structured_output with invalid JSON."""
        # Mock the response with invalid JSON
        mock_response = AIMessage(
            content="""```json
{
  "name": "John Doe",
  "age": "thirty" // This is invalid because age should be an integer
}
```"""
        )
        mock_invoke.return_value = mock_response

        # Create a model with structured output
        structured_model = self.chat.with_structured_output(
            SimpleSchema,
            method="json_mode"
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about John Doe")
        ]
        
        # This should raise a ValueError because the JSON is invalid
        with pytest.raises(ValueError):
            structured_model.invoke(messages)

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_with_structured_output_include_raw(self, mock_invoke):
        """Test with_structured_output with include_raw=True.
        
        Note: For ASI models, we use JSON mode with appropriate instructions.
        """
        # Mock the JSON response that would be returned by the model
        mock_response = AIMessage(
            content='{"name": "John Doe", "age": 30}'
        )
        mock_invoke.return_value = mock_response

        # Create a model with structured output and include_raw=True
        structured_model = self.chat.with_structured_output(
            SimpleSchema,
            method="function_calling",
            include_raw=True
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about John Doe")
        ]
        
        result = structured_model.invoke(messages)
        
        # Check that the result has raw and parsed fields
        assert "raw" in result
        assert "parsed" in result
        assert result["raw"] == mock_response
        assert isinstance(result["parsed"], SimpleSchema)
        assert result["parsed"].name == "John Doe"
        assert result["parsed"].age == 30
        assert isinstance(result["raw"], AIMessage)
        assert result["raw"].content == '{"name": "John Doe", "age": 30}'
