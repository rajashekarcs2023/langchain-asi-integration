"""
ASI API Compatibility Layer for LangChain.

This module provides compatibility functions to make ASI work with LangChain's
standard patterns for tool calling and structured output.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Union

def extract_tool_name_from_content(content: str, available_tools: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract the tool name from the content.
    
    Args:
        content: The content to extract the tool name from
        available_tools: List of available tools with their schemas
        
    Returns:
        The extracted tool name, or None if no tool name was found
    """
    # Check for tool names in backticks
    backtick_pattern = r'`([^`]+)`'
    backtick_matches = re.findall(backtick_pattern, content)
    
    for match in backtick_matches:
        # Check if the backtick content matches any tool name (with or without namespace prefix)
        for tool in available_tools:
            if isinstance(tool, dict) and "function" in tool:
                tool_name = tool["function"].get("name")
                if tool_name:
                    # Check for exact match
                    if match.lower() == tool_name.lower():
                        return tool_name
                    # Check for namespace prefixed match (e.g., default_api.get_weather)
                    if match.lower().endswith('.' + tool_name.lower()):
                        return tool_name
    
    # Check for direct mentions of tool names
    for tool in available_tools:
        if isinstance(tool, dict) and "function" in tool:
            tool_name = tool["function"].get("name")
            if tool_name:
                # Check for exact tool name mention
                if re.search(r'\b' + re.escape(tool_name) + r'\b', content, re.IGNORECASE):
                    return tool_name
                # Check for namespace prefixed match (e.g., default_api.get_weather)
                namespace_pattern = r'\b\w+\.' + re.escape(tool_name) + r'\b'
                if re.search(namespace_pattern, content, re.IGNORECASE):
                    return tool_name
    
    # If we have tools but couldn't find a match, return the first tool name as a fallback
    if available_tools and isinstance(available_tools[0], dict) and "function" in available_tools[0]:
        return available_tools[0]["function"].get("name")
    
    return None

def extract_arguments_from_content(content: str, tool_name: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract arguments for a tool from the content.
    
    Args:
        content: The content to extract arguments from
        tool_name: The name of the tool
        tool_schema: The schema of the tool
        
    Returns:
        A dictionary of arguments for the tool
    """
    arguments = {}
    
    # Get the properties from the schema
    if not tool_schema or "parameters" not in tool_schema:
        return arguments
        
    parameters = tool_schema.get("parameters", {})
    properties = parameters.get("properties", {})
    
    # Check for bullet point format (common in ASI responses)
    # Example: "- location: San Francisco, CA"
    bullet_pattern = r'-\s*([\w_]+)\s*:\s*([^\n]+)'
    bullet_matches = re.findall(bullet_pattern, content)
    
    if bullet_matches:
        for param_name, value in bullet_matches:
            param_name = param_name.strip()
            value = value.strip()
            
            # Check if this parameter exists in the schema
            if param_name in properties:
                # Convert value to the appropriate type based on the schema
                param_schema = properties[param_name]
                param_type = param_schema.get("type")
                if param_type == "number" or param_type == "integer":
                    try:
                        value = float(value) if param_type == "number" else int(value)
                    except ValueError:
                        pass
                elif param_type == "boolean":
                    value = value.lower() in ["true", "yes", "1"]
                    
                arguments[param_name] = value
    
    # If no bullet points found, try the standard pattern
    if not arguments:
        # Extract arguments using simple patterns
        for param_name, param_schema in properties.items():
            # Look for "param_name: value" pattern
            pattern = rf"{param_name}[\s]*:[\s]*([^\n,]+)"
            match = re.search(pattern, content, re.IGNORECASE)
            
            if match:
                value = match.group(1).strip()
                
                # Convert value to the appropriate type based on the schema
                param_type = param_schema.get("type")
                if param_type == "number" or param_type == "integer":
                    try:
                        value = float(value) if param_type == "number" else int(value)
                    except ValueError:
                        # If conversion fails, skip this argument
                        pass
                elif param_type == "boolean":
                    value = value.lower() in ["true", "yes", "1"]
                    
                arguments[param_name] = value
    
    # Return the arguments we were able to extract
    return arguments

def create_synthetic_tool_call(
    content: str, 
    tool_name: str, 
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a synthetic tool call in OpenAI-compatible format.
    
    Args:
        content: The content from ASI's response
        tool_name: The name of the tool
        arguments: The arguments for the tool
        
    Returns:
        A synthetic tool call in OpenAI-compatible format
    """
    return {
        "id": f"call_{str(uuid.uuid4())[:6]}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments)
        }
    }

def extract_json_from_content(content: str) -> Optional[str]:
    """
    Extract JSON from content.
    
    This function attempts to extract JSON from content, which may be wrapped in
    markdown code blocks or other formatting.
    
    Args:
        content: The content to extract JSON from
        
    Returns:
        The extracted JSON as a string, or None if no JSON could be extracted
    """
    if not content:
        return None
    
    # First, try to find the start of JSON content by looking for common patterns
    # ASI often adds text before the actual JSON
    json_start_patterns = [
        r'\[\s*\n?\s*\{',  # Start of a JSON array with objects
        r'\{\s*\n?\s*"',  # Start of a JSON object
        r'\[\s*\n?\s*"',  # Start of a JSON array with strings
    ]
    
    # Find the earliest occurrence of a JSON start pattern
    earliest_start = len(content)
    for pattern in json_start_patterns:
        matches = re.search(pattern, content)
        if matches and matches.start() < earliest_start:
            earliest_start = matches.start()
    
    # If we found a JSON start pattern, extract from that point
    if earliest_start < len(content):
        content = content[earliest_start:]
    
    # Try to extract JSON from markdown code blocks first (common ASI pattern)
    json_code_block_patterns = [
        r"```(?:json)?\s*([\s\S]*?)\s*```",  # Standard markdown code block
        r"`{1,2}([\s\S]*?)`{1,2}",  # Single or double backtick inline code
    ]
    
    for pattern in json_code_block_patterns:
        matches = re.findall(pattern, content)
        # Check each match to see if it's valid JSON
        for match in matches:
            # Clean up the match - remove leading/trailing whitespace
            cleaned_match = match.strip()
            try:
                # Validate that this is proper JSON by parsing it
                json.loads(cleaned_match)
                return cleaned_match
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON was found in code blocks, try to find JSON in the content directly
    try:
        # Check if the entire content is valid JSON
        json.loads(content)
        return content.strip()
    except json.JSONDecodeError:
        # Try to find complete JSON arrays or objects
        # This is more robust than the previous approach
        try:
            # For arrays, find balanced [ and ] pairs
            if content.strip().startswith('['):
                # Count opening and closing brackets to find the complete array
                bracket_count = 0
                for i, char in enumerate(content):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            # We've found a complete array
                            json_str = content[:i+1].strip()
                            json.loads(json_str)  # Validate it's valid JSON
                            return json_str
            
            # For objects, find balanced { and } pairs
            elif content.strip().startswith('{'):
                # Count opening and closing braces to find the complete object
                brace_count = 0
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # We've found a complete object
                            json_str = content[:i+1].strip()
                            json.loads(json_str)  # Validate it's valid JSON
                            return json_str
        except (json.JSONDecodeError, IndexError):
            pass
        
        # If we still haven't found valid JSON, try to extract objects or arrays
        object_pattern = r"(\{[\s\S]*?\})"
        array_pattern = r"(\[[\s\S]*?\])"
        
        # Check for JSON objects
        object_matches = re.findall(object_pattern, content)
        for match in object_matches:
            try:
                json.loads(match)
                return match.strip()
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                fixed_match = match
                # Replace single quotes with double quotes (common ASI issue)
                fixed_match = re.sub(r"'([^']*)'\s*:", r'"\1":', fixed_match)
                fixed_match = re.sub(r":\s*'([^']*)'([,}])", r':"\1"\2', fixed_match)
                
                try:
                    json.loads(fixed_match)
                    return fixed_match.strip()
                except json.JSONDecodeError:
                    continue
        
        # Check for JSON arrays
        array_matches = re.findall(array_pattern, content)
        for match in array_matches:
            try:
                json.loads(match)
                return match.strip()
            except json.JSONDecodeError:
                continue
    except (json.JSONDecodeError, ValueError):
        return None

def enhance_asi_response(
    response_data: Dict[str, Any],
    available_tools: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Enhance ASI's response to make it compatible with LangChain's expectations.
    
    This function modifies ASI's response to make it compatible with LangChain's
    expectations for tool calling and structured output.
    
    Args:
        response_data: The response data from ASI
        available_tools: List of available tools with their schemas
        
    Returns:
        The enhanced response data
    """
    # Make a copy of the response data to avoid modifying the original
    enhanced_data = response_data.copy()
    
    # Check if we have choices
    if "choices" not in enhanced_data or not enhanced_data["choices"]:
        return enhanced_data
    
    # Get the first choice
    choice = enhanced_data["choices"][0]
    
    # Check if we have a message
    if "message" not in choice:
        return enhanced_data
    
    message = choice["message"]
    content = message.get("content", "")
    
    # Check if this is a tool call
    finish_reason = choice.get("finish_reason")
    has_detected_tool = False
    
    # Case 1: ASI indicates tool usage via finish_reason but doesn't provide tool_calls
    if finish_reason == "tool_calls" and "tool_calls" not in message and available_tools:
        # ASI has indicated tool usage but hasn't provided a tool_calls array
        # Extract the tool name from the content
        tool_name = extract_tool_name_from_content(content, available_tools)
        
        if tool_name:
            # Find the tool schema
            tool_schema = None
            for tool in available_tools:
                if isinstance(tool, dict) and "function" in tool:
                    if tool["function"].get("name") == tool_name:
                        tool_schema = tool["function"]
                        break
            
            # Extract arguments from the content
            arguments = {}
            if tool_schema:
                arguments = extract_arguments_from_content(content, tool_name, tool_schema)
            
            # Create a synthetic tool call
            tool_call = create_synthetic_tool_call(content, tool_name, arguments)
            
            # Add the tool call to the message
            message["tool_calls"] = [tool_call]
            has_detected_tool = True
    
    # Case 2: ASI doesn't indicate tool usage via finish_reason but mentions tools in content
    elif "tool_calls" not in message and available_tools and content:
        # First, check if any tool name is directly mentioned in the content
        for tool in available_tools:
            if isinstance(tool, dict) and "function" in tool:
                tool_name = tool["function"].get("name")
                if tool_name and (f"`{tool_name}`" in content or re.search(r'\b' + re.escape(tool_name) + r'\b', content, re.IGNORECASE)):
                    # Found a tool mention, extract arguments
                    tool_schema = tool["function"]
                    arguments = extract_arguments_from_content(content, tool_name, tool_schema)
                    
                    # Create a synthetic tool call
                    tool_call = create_synthetic_tool_call(content, tool_name, arguments)
                    
                    # Add the tool call to the message
                    message["tool_calls"] = [tool_call]
                    
                    # Set finish_reason to tool_calls to match expected behavior
                    choice["finish_reason"] = "tool_calls"
                    has_detected_tool = True
                    break
        
        # If no direct tool name match, check for backtick patterns
        if not has_detected_tool:
            backtick_pattern = r'`([^`]+)`'
            backtick_matches = re.findall(backtick_pattern, content)
            
            for match in backtick_matches:
                # Check if the backtick content matches any tool name
                for tool in available_tools:
                    if isinstance(tool, dict) and "function" in tool:
                        tool_name = tool["function"].get("name")
                        if tool_name and match.lower() == tool_name.lower():
                            # Found a tool mention, extract arguments
                            tool_schema = tool["function"]
                            arguments = extract_arguments_from_content(content, tool_name, tool_schema)
                            
                            # Create a synthetic tool call
                            tool_call = create_synthetic_tool_call(content, tool_name, arguments)
                            
                            # Add the tool call to the message
                            message["tool_calls"] = [tool_call]
                            
                            # Set finish_reason to tool_calls to match expected behavior
                            choice["finish_reason"] = "tool_calls"
                            has_detected_tool = True
                            break
                    
                    if has_detected_tool:
                        break
    
    # Case 3: Check if this is a function call (for structured output)
    elif finish_reason == "function_calling" and "function_call" not in message:
        # Try to extract JSON from the content
        json_str = extract_json_from_content(content)
        
        if json_str:
            try:
                # Parse the JSON string to ensure it's valid
                json_data = json.loads(json_str)
                # Create a synthetic function call
                function_call = {
                    "name": "output_formatter",  # Generic name
                    "arguments": json_str
                }
                
                # Add the function call to the message
                message["function_call"] = function_call
            except json.JSONDecodeError:
                # If parsing fails, just return the original response
                return enhanced_data
    
    # Ensure that any tool_calls are also added to additional_kwargs
    # This is critical for LangChain compatibility
    if "tool_calls" in message and "additional_kwargs" not in message:
        message["additional_kwargs"] = {"tool_calls": message["tool_calls"]}
    elif "tool_calls" in message:
        if isinstance(message.get("additional_kwargs"), dict):
            message["additional_kwargs"]["tool_calls"] = message["tool_calls"]
        else:
            message["additional_kwargs"] = {"tool_calls": message["tool_calls"]}
    
    return enhanced_data
