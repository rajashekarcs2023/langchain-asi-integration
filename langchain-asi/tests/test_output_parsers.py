"""Tests for the ASI output parsers."""
import unittest
from unittest import mock

from langchain_asi import ASIJsonOutputParser, ASIJsonOutputParserWithValidation
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field


class TestMovie(BaseModel):
    """Test movie model."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")


class TestASIJsonOutputParser(unittest.TestCase):
    """Test ASIJsonOutputParser."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        parser = ASIJsonOutputParser()
        text = '{"name": "John", "age": 30}'
        result = parser.parse(text)
        self.assertEqual(result["name"], "John")
        self.assertEqual(result["age"], 30)

    def test_parse_json_in_markdown(self):
        """Test parsing JSON in markdown code block."""
        parser = ASIJsonOutputParser()
        text = """Here's the JSON:

```json
{"name": "John", "age": 30}
```

Hope that helps!"""
        result = parser.parse(text)
        self.assertEqual(result["name"], "John")
        self.assertEqual(result["age"], 30)

    def test_parse_json_in_text(self):
        """Test parsing JSON embedded in text."""
        parser = ASIJsonOutputParser()
        text = """Here's the information you requested: {"name": "John", "age": 30}. Let me know if you need anything else."""
        result = parser.parse(text)
        self.assertEqual(result["name"], "John")
        self.assertEqual(result["age"], 30)

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        parser = ASIJsonOutputParser()
        text = "This is not JSON"
        with self.assertRaises(OutputParserException):
            parser.parse(text)


class TestASIJsonOutputParserWithValidation(unittest.TestCase):
    """Test ASIJsonOutputParserWithValidation."""

    def test_parse_valid_json_with_validation(self):
        """Test parsing valid JSON with validation."""
        parser = ASIJsonOutputParserWithValidation(pydantic_object=TestMovie)
        text = '{"title": "The Matrix", "year": 1999}'
        result = parser.parse(text)
        self.assertEqual(result.title, "The Matrix")
        self.assertEqual(result.year, 1999)

    def test_parse_invalid_schema(self):
        """Test parsing JSON that doesn't match the schema."""
        parser = ASIJsonOutputParserWithValidation(pydantic_object=TestMovie)
        text = '{"name": "John", "age": 30}'
        with self.assertRaises(OutputParserException):
            parser.parse(text)

    def test_parse_json_in_markdown_with_validation(self):
        """Test parsing JSON in markdown with validation."""
        parser = ASIJsonOutputParserWithValidation(pydantic_object=TestMovie)
        text = """Here's the JSON:

```json
{"title": "The Matrix", "year": 1999}
```

Hope that helps!"""
        result = parser.parse(text)
        self.assertEqual(result.title, "The Matrix")
        self.assertEqual(result.year, 1999)


if __name__ == "__main__":
    unittest.main()
