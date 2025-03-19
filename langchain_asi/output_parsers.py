"""Output parsers for ASI integrations."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, field_validator
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


class ASIJsonOutputParser(JsonOutputParser):
    """Output parser for ASI JSON responses.
    
    This parser handles JSON responses from ASI models, with special handling for
    cases where the model might not return valid JSON.
    
    Example:
        .. code-block:: python
            
            from langchain_asi import ASI1ChatModel, ASIJsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            
            model = ASI1ChatModel()
            prompt = ChatPromptTemplate.from_template(
                "Generate a JSON object with keys 'name' and 'age' for {person}."
            )
            parser = ASIJsonOutputParser()
            
            chain = prompt | model | parser
            result = chain.invoke({"person": "John Doe"})
    """
    
    pydantic_object: Optional[Type[BaseModel]] = None
    """Optional pydantic object to parse the JSON into."""
    
    def parse(self, text: str) -> Union[Dict, List]:
        """Parse the output text into a JSON object.
        
        Args:
            text: Text to parse into a JSON object.
            
        Returns:
            Parsed JSON object as a Python dictionary or list.
            
        Raises:
            OutputParserException: If the text cannot be parsed into a JSON object.
        """
        # Try to extract JSON from the text using regex
        json_match = re.search(r"```json\s*(.+?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        else:
            # Look for JSON objects within the text
            json_match = re.search(r"\{.+\}", text, re.DOTALL)
            if json_match:
                text = json_match.group(0).strip()
        
        try:
            json_object = json.loads(text)
            
            # If a pydantic model is provided, validate the JSON against it
            if self.pydantic_object is not None:
                return self.pydantic_object.model_validate(json_object)
                
            return json_object
        except json.JSONDecodeError as e:
            raise OutputParserException(
                f"Failed to parse ASI output as JSON: {e}\n\nText: {text}"
            )


class ASIJsonResponseModel(BaseModel):
    """Pydantic model for ASI JSON responses."""
    
    name: str
    age: int
    
    @field_validator('name')
    def name_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Name must not be empty')
        return v
    
    @field_validator('age')
    def age_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Age must be a positive integer')
        return v


class ASIJsonOutputParserWithValidation(ASIJsonOutputParser):
    """Output parser for ASI JSON responses with validation."""
    
    pydantic_object: Type[BaseModel] = Field(default=ASIJsonResponseModel)
    
    def parse(self, text: str) -> Any:
        """Parse the output text into a validated Pydantic object.
        
        Args:
            text: Text to parse into a JSON object.
            
        Returns:
            Validated Pydantic object.
            
        Raises:
            OutputParserException: If the text cannot be parsed into a JSON object
                or if the JSON object does not match the Pydantic model.
        """
        try:
            # First parse the text into a JSON object
            json_object = super().parse(text)
            
            # Then validate the JSON object against the Pydantic model
            if self.pydantic_object is not None:
                return self.pydantic_object.model_validate(json_object)
            
            return json_object
        except Exception as e:
            raise OutputParserException(
                f"Failed to parse ASI output as {self.pydantic_object.__name__}: {e}\n\nText: {text}"
            )