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
            else:
                # Try to extract Python dict from Python code blocks
                python_match = re.search(r"```python\s*(.+?)\s*```", text, re.DOTALL)
                if python_match:
                    python_code = python_match.group(1).strip()
                    # Look for dictionary definitions
                    dict_match = re.search(r"\{[^\{\}]*?\"([^\"]+)\"\s*:\s*[^\{\}]+\}", python_code, re.DOTALL)
                    if dict_match:
                        text = dict_match.group(0).strip()
                    # Look for function calls with keyword arguments
                    else:
                        func_call_match = re.search(r"\w+\s*\(\s*([^\(\)]+)\s*\)", python_code, re.DOTALL)
                        if func_call_match:
                            args_text = func_call_match.group(1).strip()
                            # Convert keyword arguments to a JSON-like format
                            args_dict = {}
                            for arg_match in re.finditer(r"(\w+)\s*=\s*([^,]+)(?:,|$)", args_text):
                                key = arg_match.group(1).strip()
                                value = arg_match.group(2).strip()
                                # Remove quotes from string values
                                if (value.startswith('"') and value.endswith('"')) or (value.startswith('\'') and value.endswith('\'')):
                                    value = value[1:-1]
                                args_dict[key] = value
                            if args_dict:
                                text = json.dumps(args_dict)
        
        try:
            json_object = json.loads(text)
            
            # If a pydantic model is provided, validate the JSON against it
            if self.pydantic_object is not None:
                # Try to convert string values to the expected types based on schema
                schema = self.pydantic_object.model_json_schema()
                properties = schema.get("properties", {})
                
                for field_name, field_info in properties.items():
                    if field_name in json_object:
                        field_type = field_info.get("type")
                        field_value = json_object[field_name]
                        
                        # Convert string numbers to actual numbers
                        if field_type == "number" and isinstance(field_value, str):
                            try:
                                # Extract first number from string
                                number_match = re.search(r"(\d+(?:\.\d+)?)", field_value)
                                if number_match:
                                    json_object[field_name] = float(number_match.group(1))
                            except (ValueError, TypeError):
                                pass
                        
                        # Convert string booleans to actual booleans
                        if field_type == "boolean" and isinstance(field_value, str):
                            lower_value = field_value.lower()
                            if lower_value in ["yes", "true"]:
                                json_object[field_name] = True
                            elif lower_value in ["no", "false"]:
                                json_object[field_name] = False
                
                return self.pydantic_object.model_validate(json_object)
                
            return json_object
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract structured data from the text
            if self.pydantic_object is not None:
                try:
                    # Extract key-value pairs from the text using a general pattern
                    extracted_data = {}
                    schema = self.pydantic_object.model_json_schema()
                    properties = schema.get("properties", {})
                    
                    # Look for patterns like "Field: Value" in the text
                    for field_name in properties.keys():
                        pattern = fr"(?:^|\n)\s*{field_name}\s*:?\s*([^\n]+)"
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            extracted_data[field_name] = match.group(1).strip()
                    
                    # If we found any fields, return them
                    if extracted_data:
                        return extracted_data
                except Exception:
                    pass
            
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