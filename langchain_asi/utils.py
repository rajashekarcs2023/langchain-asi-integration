"""Utility functions for ASI integrations."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type, Union, Sequence, Callable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


def prepare_tools(
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
) -> List[Dict[str, Any]]:
    """Convert various tool formats to the format expected by the ASI API.
    
    Args:
        tools: A list of tools in different formats.
        
    Returns:
        A list of tool dictionaries in the format expected by the ASI API.
    """
    prepared_tools = []
    
    for tool in tools:
        if isinstance(tool, dict):
            # If it's already a dictionary, assume it's in the right format
            prepared_tools.append(tool)
        else:
            # Otherwise convert it to the OpenAI tool format
            prepared_tools.append(convert_to_openai_tool(tool))
            
    return prepared_tools