import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, Union, Sequence, Callable
from pydantic import ValidationError

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_asi.chat_models import ChatASI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAICompatibleASI:
    """Adapter class to make ASI work more like OpenAI with LangGraph."""
    
    def __init__(self, model: ChatASI):
        """Initialize the adapter with an ASI model."""
        self.model = model
        self.tools = []
        self._tool_choice = None
        self._structured_output_schema = None
        self._schema_fields_info = None
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], BaseTool, Callable]],
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Any:
        """Bind tools to the model in OpenAI-compatible format."""
        # Store the tools for later use
        self.tools = tools
        self._tool_choice = tool_choice
        
        # Return self for method chaining
        return self
    
    def bind_functions(
        self,
        functions: List[Dict[str, Any]],
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Any:
        """Bind functions to the model in OpenAI-compatible format.
        
        This method is similar to bind_tools but follows the OpenAI naming convention
        for better compatibility with existing code that uses OpenAI's API.
        
        Args:
            functions: List of function definitions in OpenAI format
            function_call: Optional specification of which function to call
            
        Returns:
            Self for method chaining
        """
        # Convert functions to tools format
        tools = []
        for func in functions:
            tools.append({
                "type": "function",
                "function": func
            })
        
        # Convert function_call to tool_choice format if needed
        tool_choice = None
        if function_call:
            if isinstance(function_call, str):
                tool_choice = {
                    "type": "function",
                    "function": {"name": function_call}
                }
            else:
                tool_choice = {
                    "type": "function",
                    "function": function_call
                }
        
        # Use bind_tools to store the tools
        return self.bind_tools(tools, tool_choice)
    
    def _parse_with_fallback(
        self, text: str, pydantic_schema: Any
    ) -> Any:
        """Parse the output with fallback methods if the primary method fails."""
        # First try to parse as a JSON object
        try:
            # Check if the text is wrapped in a code block
            if "```json" in text and "```" in text.split("```json", 1)[1]:
                # Extract the JSON from the code block
                json_str = text.split("```json", 1)[1].split("```", 1)[0].strip()
                parsed_json = json.loads(json_str)
                return pydantic_schema.parse_obj(parsed_json)
            elif "```" in text and "```" in text.split("```", 1)[1]:
                # Extract from a generic code block
                json_str = text.split("```", 1)[1].split("```", 1)[0].strip()
                try:
                    parsed_json = json.loads(json_str)
                    return pydantic_schema.parse_obj(parsed_json)
                except json.JSONDecodeError:
                    # Not a valid JSON in code block, continue to next method
                    pass
            
            # Try to parse the entire text as JSON
            parsed_json = json.loads(text)
            return pydantic_schema.parse_obj(parsed_json)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Failed to parse as JSON: {e}")
        
        # Try to extract JSON from the text using regex
        try:
            # Look for JSON-like patterns
            json_match = re.search(r'\{[^{}]*\}', text)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
                return pydantic_schema.parse_obj(parsed_json)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Failed to extract JSON with regex: {e}")
        
        # If all parsing attempts fail, create a default object with required fields
        try:
            # Get the required fields from the schema
            if hasattr(pydantic_schema, "__fields__"):  # Pydantic v1
                required_fields = {
                    name: field
                    for name, field in pydantic_schema.__fields__.items()
                    if field.required
                }
            else:  # Pydantic v2
                required_fields = {
                    name: field
                    for name, field in pydantic_schema.model_fields.items()
                    if field.is_required()
                }
            
            # Create a minimal valid object
            default_values = {}
            for name, field in required_fields.items():
                # Set default values based on field type
                if field.type_ == str or getattr(field, "annotation", None) == str:
                    # For string fields, use the text or a default
                    default_values[name] = text[:100] if name == "reasoning" else "default"
                elif field.type_ == int or getattr(field, "annotation", None) == int:
                    default_values[name] = 0
                elif field.type_ == bool or getattr(field, "annotation", None) == bool:
                    default_values[name] = False
                elif field.type_ == list or getattr(field, "annotation", None) == list:
                    default_values[name] = []
                elif hasattr(field, "type_") and hasattr(field.type_, "__origin__") and field.type_.__origin__ == Literal:
                    # For Literal fields, use the first allowed value
                    allowed_values = field.type_.__args__
                    default_values[name] = allowed_values[0] if allowed_values else None
                elif hasattr(field, "annotation") and hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == Literal:
                    # For Literal fields in Pydantic v2
                    allowed_values = field.annotation.__args__
                    default_values[name] = allowed_values[0] if allowed_values else None
            
            # Create the object with default values
            return pydantic_schema.parse_obj(default_values)
        except Exception as e:
            logger.error(f"Error creating default structured output: {e}")
            raise ValueError(f"Failed to parse structured output: {text}")
    
    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        """Invoke the model with OpenAI-compatible formatting."""
        try:
            # If we have tools, bind them to the model
            if self.tools and not kwargs.get("tools"):
                # Convert tools to ASI format if needed
                asi_tools = []
                for tool in self.tools:
                    if isinstance(tool, dict) and "function" in tool:
                        asi_tools.append(tool)
                    elif isinstance(tool, dict) and "type" in tool and tool["type"] == "function":
                        asi_tools.append(tool)
                    elif isinstance(tool, BaseTool):
                        asi_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.args_schema.schema() if hasattr(tool, "args_schema") else {}
                            }
                        })
                    elif hasattr(tool, 'schema') and callable(getattr(tool, 'schema')):
                        # Handle Pydantic models
                        schema = tool.schema()
                        asi_tools.append({
                            "type": "function",
                            "function": {
                                "name": schema.get('title', 'unnamed_tool'),
                                "description": schema.get('description', ''),
                                "parameters": schema
                            }
                        })
                    else:
                        # Try to convert using LangChain's utility
                        try:
                            from langchain_core.utils.function_calling import convert_to_openai_tool
                            asi_tools.append(convert_to_openai_tool(tool))
                        except Exception as e:
                            logger.warning(f"Could not convert tool to ASI format: {e}")
                            # Add a minimal tool definition as fallback
                            asi_tools.append({
                                "type": "function",
                                "function": {
                                    "name": getattr(tool, "__name__", "unnamed_tool"),
                                    "description": getattr(tool, "__doc__", ""),
                                    "parameters": {}
                                }
                            })
                
                # Apply the tools to the model
                model_with_tools = self.model.bind_tools(
                    asi_tools,
                    tool_choice=self._tool_choice
                )
                
                # Invoke the model with tools
                response = model_with_tools.invoke(messages, **kwargs)
                
                # Return the response
                return response
            
            # If we have a structured output schema, use it
            if self._structured_output_schema and not kwargs.get("structured_output"):
                # Create a model with structured output
                model_with_schema = self.model.with_structured_output(
                    self._structured_output_schema
                )
                
                # Invoke the model with structured output
                response = model_with_schema.invoke(messages, **kwargs)
                
                # Return the response
                return response
            
            # Otherwise, just invoke the model normally
            return self.model.invoke(messages, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in OpenAICompatibleASI.invoke: {e}")
            # Return a minimal AIMessage to prevent further errors
            return AIMessage(content=f"Error invoking model: {str(e)}")
            return self._parse_with_fallback(response.content, self._structured_output_schema)
        else:
            # Return the response as is
            return response
    
    def with_structured_output(self, schema: Type[BaseModel]) -> Any:
        """Configure the model to return structured output."""
        # Store the schema for later use
        self._structured_output_schema = schema
        
        # Extract field info for better prompting
        if hasattr(schema, 'schema'):
            schema_dict = schema.schema()
            self._schema_fields_info = schema_dict.get('properties', {})
        
        # Bind the schema as a tool
        return self.bind_tools([schema])
    
    def create_supervisor(
        self,
        system_prompt: str,
        members: List[str],
        schema: Type[BaseModel],
    ) -> Callable:
        """Create a supervisor function for LangGraph.
        
        Args:
            system_prompt: The system prompt for the supervisor
            members: List of team member names
            schema: Pydantic schema for the routing decision
            
        Returns:
            A supervisor function that can be used in a LangGraph
        """
        # Enhance the system prompt with additional instructions
        enhanced_system_prompt = system_prompt + f"""
        
        Think step by step:
        1. What specific information is needed to fully answer the query?
        2. Which agent is best suited to find each piece of information?
        3. What order of operations would give the most comprehensive answer?
        4. Have we gathered all necessary information to FINISH?
        
        Available team members: {', '.join(members)}
        
        IMPORTANT: You MUST use the route function to provide your response. Do not respond in plain text.
        """
        
        # Create the function definition
        function_def = {
            "name": "route",
            "description": "Select the next role based on query analysis.",
            "parameters": schema.schema()
        }
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", enhanced_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create a more robust parser with fallback mechanisms
        def parse_with_fallback(output):
            try:
                # Check if we have tool calls in the response
                if hasattr(output, 'tool_calls') and output.tool_calls:
                    # Extract the first tool call's arguments
                    tool_call = output.tool_calls[0]
                    if tool_call['name'] == 'route':
                        logger.info(f"Found tool call: {tool_call['name']}")
                        logger.debug(f"Arguments: {tool_call['args']}")
                        return tool_call['args']
                
                # If no tool calls, try to parse as JSON
                if hasattr(output, 'content') and output.content:
                    logger.info(f"Trying to parse content as JSON")
                    try:
                        # Try to extract JSON from the content
                        json_match = re.search(r'```json\n(.+?)\n```', output.content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            return json.loads(json_str)
                        else:
                            # Try to parse the whole content as JSON
                            return json.loads(output.content)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse as JSON")
                
                # If we get here, we couldn't parse the output
                logger.warning(f"Couldn't parse output: {output}")
                
                # Default to the first member as a fallback
                return {
                    "next": members[0],
                    "reasoning": "Fallback due to parsing error",
                    "information_needed": ["Initial information"]
                }
                
            except Exception as e:
                logger.error(f"Error parsing output: {e}")
                
                # Default to the first member as a fallback
                return {
                    "next": members[0],
                    "reasoning": "Fallback due to parsing error",
                    "information_needed": ["Initial information"]
                }
        
        # Create the supervisor function
        def supervisor(state):
            # Get the messages from the state
            messages = state.get("messages", [])
            
            try:
                # Create a model with the route function
                model_with_function = self.model.bind_tools(
                    [{
                        "type": "function",
                        "function": function_def
                    }],
                    tool_choice={
                        "type": "function",
                        "function": {"name": "route"}
                    }
                )
                
                # Invoke the model
                response = model_with_function.invoke(messages)
                
                # Parse the response
                result = parse_with_fallback(response)
                
                # Make sure the result has the expected fields
                if not isinstance(result, dict):
                    logger.warning(f"Unexpected result type: {type(result)}")
                    result = {
                        "next": members[0],
                        "reasoning": "Fallback due to unexpected result type",
                        "information_needed": ["Initial information"]
                    }
                
                return result
            except Exception as e:
                logger.error(f"Error in supervisor: {e}")
                # Return a fallback result
                return {
                    "next": members[0],
                    "reasoning": f"Fallback due to error: {str(e)}",
                    "information_needed": ["Initial information"]
                }
        
        return supervisor


# Helper functions for LangGraph compatibility
def create_openai_compatible_model(model: ChatASI) -> OpenAICompatibleASI:
    """Create an OpenAI-compatible model from an ASI model.
    
    This is the main entry point for using ASI with LangGraph.
    
    Args:
        model: The ASI model to adapt
    
    Returns:
        An OpenAI-compatible model
    """
    return OpenAICompatibleASI(model)
