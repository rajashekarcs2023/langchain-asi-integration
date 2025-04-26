"""
Tool extraction utilities for ASI responses.

This module provides functions to extract tool calls from natural language content
when ASI indicates tool usage but doesn't provide a structured tool_calls array.
"""

import re
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def extract_tool_calls_from_content(
    content: str, 
    available_tools: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Extract tool calls from natural language content.
    
    This function analyzes natural language content to identify tool usage patterns
    and extracts structured tool calls that match the OpenAI format.
    
    Args:
        content: The natural language content to analyze
        available_tools: List of available tools with their schemas
        
    Returns:
        A list of extracted tool calls in OpenAI-compatible format
    """
    if not content:
        return []
        
    # Initialize result
    tool_calls = []
    
    # If no tools are available, we can't extract specific tool information
    if not available_tools:
        # Try to extract any tool-like mentions
        return extract_generic_tool_calls(content)
    
    # Extract tool calls for each available tool
    for tool in available_tools:
        # Skip tools without proper structure
        if not isinstance(tool, dict) or "function" not in tool:
            continue
            
        function_info = tool["function"]
        tool_name = function_info.get("name", "")
        
        # Skip tools without names
        if not tool_name:
            continue
            
        # Check if this tool is mentioned in the content
        tool_patterns = [
            f"`{tool_name}`",  # Tool name in backticks
            f"'{tool_name}'",  # Tool name in single quotes
            f"\"{tool_name}\"",  # Tool name in double quotes
            f"\\b{tool_name}\\b",  # Tool name as a word boundary
        ]
        
        # Use a more precise pattern matching to avoid false positives
        # Look for phrases like "use the tool_name tool" or "calling tool_name"
        precise_patterns = [
            rf"use\s+(?:the\s+)?(?:`|'|\"|\b){tool_name}(?:`|'|\"|\b)",
            rf"call(?:ing)?\s+(?:the\s+)?(?:`|'|\"|\b){tool_name}(?:`|'|\"|\b)",
            rf"(?:`|'|\"|\b){tool_name}(?:`|'|\"|\b)\s+tool",
            rf"(?:`|'|\"|\b){tool_name}(?:`|'|\"|\b)\s+function",
        ]
        
        # First try precise patterns
        tool_mentioned = any(re.search(pattern, content, re.IGNORECASE) for pattern in precise_patterns)
        
        # If not found with precise patterns, fall back to simpler patterns
        if not tool_mentioned:
            tool_mentioned = any(re.search(pattern, content, re.IGNORECASE) for pattern in tool_patterns)
        
        if tool_mentioned:
            # Extract arguments for this tool
            arguments = extract_arguments_for_tool(content, function_info)
            
            # Create a tool call in OpenAI format
            tool_call = {
                "id": f"call_{str(uuid.uuid4())[:6]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments)
                }
            }
            
            tool_calls.append(tool_call)
    
    # If no specific tools were extracted but tool usage is indicated,
    # try to extract generic tool calls
    if not tool_calls and ("tool" in content.lower() or "function" in content.lower()):
        generic_calls = extract_generic_tool_calls(content)
        
        # Filter out any generic calls that don't match available tool names
        if available_tools:
            available_tool_names = [
                tool.get("function", {}).get("name", "") 
                for tool in available_tools 
                if isinstance(tool, dict) and "function" in tool
            ]
            
            generic_calls = [
                call for call in generic_calls 
                if call.get("function", {}).get("name", "") in available_tool_names
            ]
        
        return generic_calls
    
    return tool_calls

def extract_generic_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    Extract generic tool calls when specific tool information is not available.
    
    Args:
        content: The natural language content to analyze
        
    Returns:
        A list of extracted generic tool calls
    """
    tool_calls = []
    
    # Pattern to find tool/function names in various formats
    # Looks for:
    # 1. Words after "use the" followed by tool/function in backticks, quotes, etc.
    # 2. Words after "calling" followed by tool/function
    # 3. Words inside backticks that look like function names
    tool_name_patterns = [
        r"use\s+the\s+[`'\"]?(\w+)[`'\"]?\s+(?:tool|function)",  # "use the get_weather tool"
        r"call(?:ing)?\s+(?:the\s+)?[`'\"]?(\w+)[`'\"]?",  # "calling get_weather"
        r"`(\w+)`",  # "`get_weather`"
        r"function\s+[`'\"]?(\w+)[`'\"]?",  # "function get_weather"
        r"tool\s+[`'\"]?(\w+)[`'\"]?",  # "tool get_weather"
    ]
    
    # Try to extract tool names
    tool_names = []
    for pattern in tool_name_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        tool_names.extend(matches)
    
    # Remove duplicates and empty strings
    tool_names = [name for name in set(tool_names) if name]
    
    # Filter out common words that aren't likely to be tool names
    common_words = ["with", "the", "and", "or", "for", "to", "in", "on", "at", "by", "from"]
    tool_names = [name for name in tool_names if name.lower() not in common_words]
    
    # For each potential tool name, try to extract arguments
    for tool_name in tool_names:
        # Try to extract arguments for this tool name
        arguments = {}
        
        # Look for JSON-like structures
        json_pattern = r"\{([^{}]*)\}"
        json_matches = re.findall(json_pattern, content)
        
        if json_matches:
            # Try to parse each match as JSON
            for json_str in json_matches:
                # Add braces back
                json_str = "{" + json_str + "}"
                try:
                    args = json.loads(json_str)
                    if isinstance(args, dict):
                        arguments = args
                        break
                except json.JSONDecodeError:
                    continue
        
        # If no JSON found, look for key-value pairs in the text
        if not arguments:
            # Pattern for "key: value" or "key = value"
            kv_pattern = r"(\w+)[\s]*[:=][\s]*['\"]?([^,'\"]*)['\")?"
            kv_matches = re.findall(kv_pattern, content)
            
            if kv_matches:
                arguments = {k.strip(): v.strip() for k, v in kv_matches}
            
            # Look for specific parameter mentions near the tool name
            tool_context = get_tool_context(content, tool_name)
            if tool_context:
                # Look for location='...' or location="..." patterns
                location_pattern = r"location\s*=\s*['\"]([^'\"]+)['\"]"  
                location_match = re.search(location_pattern, tool_context)
                if location_match:
                    arguments["location"] = location_match.group(1)
                
                # Look for unit='...' or unit="..." patterns
                unit_pattern = r"unit\s*=\s*['\"]([^'\"]+)['\"]"  
                unit_match = re.search(unit_pattern, tool_context)
                if unit_match:
                    arguments["unit"] = unit_match.group(1)
        
        # Create a tool call
        tool_call = {
            "id": f"call_{str(uuid.uuid4())[:6]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments)
            }
        }
        
        tool_calls.append(tool_call)
    
    return tool_calls


def get_tool_context(content: str, tool_name: str) -> Optional[str]:
    """
    Extract the context around a tool name mention.
    
    Args:
        content: The content to search in
        tool_name: The tool name to find context for
        
    Returns:
        The context around the tool name, or None if not found
    """
    # Find the position of the tool name
    tool_pos = content.lower().find(tool_name.lower())
    if tool_pos == -1:
        # Try with backticks
        tool_pos = content.lower().find(f"`{tool_name.lower()}`")
        if tool_pos == -1:
            return None
    
    # Extract a window of text around the tool name
    start_pos = max(0, tool_pos - 50)
    end_pos = min(len(content), tool_pos + len(tool_name) + 100)
    
    return content[start_pos:end_pos]

def extract_arguments_for_tool(content: str, function_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract arguments for a specific tool based on its schema.
    
    Args:
        content: The natural language content to analyze
        function_info: The function schema information
        
    Returns:
        A dictionary of extracted arguments
    """
    arguments = {}
    
    # Get parameter information from the schema
    parameters = function_info.get("parameters", {})
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    
    # For each property in the schema, try to extract its value
    for param_name, param_info in properties.items():
        param_type = param_info.get("type")
        param_description = param_info.get("description", "")
        
        # Try different extraction strategies based on parameter type
        if param_type == "string":
            value = extract_string_parameter(content, param_name, param_description)
            if value:
                arguments[param_name] = value
        elif param_type == "number" or param_type == "integer":
            value = extract_numeric_parameter(content, param_name, param_description)
            if value is not None:
                arguments[param_name] = value
        elif param_type == "boolean":
            value = extract_boolean_parameter(content, param_name, param_description)
            if value is not None:
                arguments[param_name] = value
        elif param_type == "array":
            value = extract_array_parameter(content, param_name, param_description, param_info)
            if value:
                arguments[param_name] = value
        elif param_type == "object":
            value = extract_object_parameter(content, param_name, param_description, param_info)
            if value:
                arguments[param_name] = value
    
    # Ensure all required parameters are included
    for req_param in required:
        if req_param not in arguments:
            # Try harder to extract required parameters
            param_info = properties.get(req_param, {})
            param_type = param_info.get("type")
            
            if param_type == "string":
                # For required string parameters, use a more aggressive extraction
                # that might be less accurate but ensures we have a value
                patterns = [
                    f"{req_param}\\s*[:=]\\s*['\"]?([^'\"]+)['\"]?",
                    f"{req_param}\\s+(?:is|as)\\s+['\"]?([^'\"]+)['\"]?",
                    f"['\"]?([^'\"]+)['\"]?\\s+(?:for|as)\\s+{req_param}",
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        arguments[req_param] = match.group(1).strip()
                        break
            
            # If still not found, provide a default value based on type
            if req_param not in arguments:
                if param_type == "string":
                    arguments[req_param] = ""
                elif param_type == "number" or param_type == "integer":
                    arguments[req_param] = 0
                elif param_type == "boolean":
                    arguments[req_param] = False
                elif param_type == "array":
                    arguments[req_param] = []
                elif param_type == "object":
                    arguments[req_param] = {}
    
    return arguments

def extract_string_parameter(content: str, param_name: str, param_description: str) -> Optional[str]:
    """Extract a string parameter from content."""
    # Pattern for "param_name: value" or "param_name = value"
    patterns = [
        f"{param_name}\\s*[:=]\\s*['\"]?([^'\"\\s][^'\"]*?)['\"]?[,\\s]",
        f"{param_name}\\s+(?:is|as)\\s+['\"]?([^'\"]+)['\"]?",
        f"['\"]?([^'\"]+)['\"]?\\s+(?:for|as)\\s+{param_name}",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If not found by name, try using the description
    if param_description:
        # Create description keywords
        keywords = [word.lower() for word in param_description.split() if len(word) > 3]
        
        # Look for quoted strings near description keywords
        for keyword in keywords:
            if keyword in content.lower():
                # Find quoted strings near this keyword
                context = content[max(0, content.lower().find(keyword) - 50):
                                 min(len(content), content.lower().find(keyword) + 50)]
                quoted = re.findall(r"['\"]([^'\"]+)['\"]", context)
                
                if quoted:
                    return quoted[0]
    
    return None

def extract_numeric_parameter(content: str, param_name: str, param_description: str) -> Optional[float]:
    """Extract a numeric parameter from content."""
    # Pattern for "param_name: 123" or "param_name = 123"
    patterns = [
        f"{param_name}\\s*[:=]\\s*([0-9]+\\.?[0-9]*)",
        f"{param_name}\\s+(?:is|as)\\s+([0-9]+\\.?[0-9]*)",
        f"([0-9]+\\.?[0-9]*)\\s+(?:for|as)\\s+{param_name}",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # If not found by name, try using the description
    if param_description:
        # Create description keywords
        keywords = [word.lower() for word in param_description.split() if len(word) > 3]
        
        # Look for numbers near description keywords
        for keyword in keywords:
            if keyword in content.lower():
                # Find numbers near this keyword
                context = content[max(0, content.lower().find(keyword) - 50):
                                 min(len(content), content.lower().find(keyword) + 50)]
                numbers = re.findall(r"([0-9]+\\.?[0-9]*)", context)
                
                if numbers:
                    try:
                        return float(numbers[0])
                    except ValueError:
                        continue
    
    return None

def extract_boolean_parameter(content: str, param_name: str, param_description: str) -> Optional[bool]:
    """Extract a boolean parameter from content."""
    # Pattern for "param_name: true" or "param_name = false"
    patterns = [
        f"{param_name}\\s*[:=]\\s*(true|false|yes|no)",
        f"{param_name}\\s+(?:is|as)\\s+(true|false|yes|no)",
        f"(true|false|yes|no)\\s+(?:for|as)\\s+{param_name}",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            return value in ["true", "yes"]
    
    # Check for positive/negative sentiment around the parameter name
    param_context_pattern = f"[^.]*{param_name}[^.]*\\."
    param_contexts = re.findall(param_context_pattern, content, re.IGNORECASE)
    
    for context in param_contexts:
        # Check for positive indicators
        positive_indicators = ["yes", "true", "recommend", "positive", "good", "great", "excellent"]
        negative_indicators = ["no", "false", "don't recommend", "negative", "bad", "poor"]
        
        if any(indicator in context.lower() for indicator in positive_indicators):
            return True
        if any(indicator in context.lower() for indicator in negative_indicators):
            return False
    
    return None

def extract_array_parameter(content: str, param_name: str, param_description: str, param_info: Dict[str, Any]) -> Optional[List[Any]]:
    """Extract an array parameter from content."""
    # Try to find JSON arrays
    array_patterns = [
        f"{param_name}\\s*[:=]\\s*(\\[.*?\\])",
        f"{param_name}\\s+(?:is|as)\\s+(\\[.*?\\])",
    ]
    
    for pattern in array_patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # If no JSON array found, try to extract a list from natural language
    items_type = param_info.get("items", {}).get("type", "string")
    
    # Look for lists in various formats
    list_patterns = [
        # Numbered list: 1. item1 2. item2
        r"\d+\.\s*([^\\d\\.]+)(?=\\d+\\.|$)",
        # Bullet list: • item1 • item2
        r"[•\\*-]\\s*([^•\\*-]+)(?=[•\\*-]|$)",
        # Comma-separated list: item1, item2, item3
        f"{param_name}\\s*[:=]\\s*([^\\[\\]]+?)(?=\\.|$)",
    ]
    
    for pattern in list_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        if matches:
            # Convert matches to the appropriate type
            if items_type == "string":
                return [item.strip() for item in matches]
            elif items_type == "number" or items_type == "integer":
                result = []
                for item in matches:
                    try:
                        result.append(float(item.strip()))
                    except ValueError:
                        # Skip non-numeric items
                        continue
                return result if result else None
            elif items_type == "boolean":
                result = []
                for item in matches:
                    lower_item = item.strip().lower()
                    if lower_item in ["true", "yes"]:
                        result.append(True)
                    elif lower_item in ["false", "no"]:
                        result.append(False)
                return result if result else None
    
    # If comma-separated list not found by pattern, try a more general approach
    if param_name in content:
        # Find the part after the parameter name
        parts = content.split(param_name)
        if len(parts) > 1:
            after_param = parts[1]
            # Find the first sentence or section
            end_idx = min(idx for idx in [after_param.find("."), after_param.find("\n")] if idx > 0)
            if end_idx > 0:
                section = after_param[:end_idx]
                # Split by commas and convert
                items = [item.strip() for item in section.split(",")]
                if items_type == "string":
                    return items
                elif items_type == "number" or items_type == "integer":
                    result = []
                    for item in items:
                        try:
                            result.append(float(item))
                        except ValueError:
                            # Skip non-numeric items
                            continue
                    return result if result else None
    
    return None

def extract_object_parameter(content: str, param_name: str, param_description: str, param_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract an object parameter from content."""
    # Try to find JSON objects
    object_patterns = [
        f"{param_name}\\s*[:=]\\s*(\\{{.*?\\}})",
        f"{param_name}\\s+(?:is|as)\\s+(\\{{.*?\\}})",
    ]
    
    for pattern in object_patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # If no JSON object found, try to extract properties individually
    result = {}
    properties = param_info.get("properties", {})
    
    for prop_name, prop_info in properties.items():
        prop_type = prop_info.get("type")
        prop_description = prop_info.get("description", "")
        
        # Use the appropriate extraction method based on property type
        if prop_type == "string":
            value = extract_string_parameter(content, prop_name, prop_description)
            if value:
                result[prop_name] = value
        elif prop_type == "number" or prop_type == "integer":
            value = extract_numeric_parameter(content, prop_name, prop_description)
            if value is not None:
                result[prop_name] = value
        elif prop_type == "boolean":
            value = extract_boolean_parameter(content, prop_name, prop_description)
            if value is not None:
                result[prop_name] = value
    
    return result if result else None
