


def _create_chat_result(self, response_data):
    """Create a ChatResult from the response data.

    Args:
        response_data: The response data from the API.

    Returns:
        A ChatResult containing the generated responses.
    """
    # Check for error responses
    if "error" in response_data:
        error_message = response_data.get("error", {}).get("message", "Unknown error")
        raise ValueError(f"API error: {error_message}")
        
    if "choices" not in response_data:
        raise ValueError(f"Invalid response format: {response_data}")

    message_data = response_data["choices"][0]["message"]
    content = message_data.get("content", "")
    finish_reason = response_data["choices"][0].get("finish_reason", None)
    
    # Extract additional information if available
    additional_kwargs = {}
    
    # Copy any existing additional kwargs from the message
    for key, value in message_data.items():
        if key not in ["role", "content"]:
            additional_kwargs[key] = value
    
    # Handle tool calls - either explicit or implicit
    tool_calls = []
    
    # Case 1: Explicit tool_calls in the message
    if "tool_calls" in message_data:
        try:
            # Format tool_calls to match the expected format for LangChain
            raw_tool_calls = message_data["tool_calls"]
            
            # Handle case where tool_calls might not be a list
            if not isinstance(raw_tool_calls, list):
                if isinstance(raw_tool_calls, dict):
                    raw_tool_calls = [raw_tool_calls]
                else:
                    logger.warning(f"Unexpected tool_calls format: {raw_tool_calls}")
                    raw_tool_calls = []
            
            for tool_call in raw_tool_calls:
                try:
                    # Generate a tool ID if not present
                    tool_id = tool_call.get("id")
                    if not tool_id:
                        tool_id = f"call_{str(uuid.uuid4())[:6]}"
                    
                    # Handle different tool call formats
                    if "function" in tool_call:
                        # Standard format with function field
                        function_data = tool_call["function"]
                        
                        # Ensure arguments is a valid JSON string
                        arguments = function_data.get("arguments", "{}")
                        if not isinstance(arguments, str):
                            # Convert dict to JSON string if needed
                            try:
                                arguments = json.dumps(arguments)
                            except (TypeError, ValueError):
                                arguments = "{}"
                        
                        formatted_tool_call = {
                            "id": tool_id,
                            "type": "function",  # Use 'function' to match OpenAI format
                            "function": {
                                "name": function_data.get("name", ""),
                                "arguments": arguments
                            }
                        }
                        tool_calls.append(formatted_tool_call)
                    elif "name" in tool_call:
                        # ASI might use a flatter structure
                        # Ensure arguments is a valid JSON string
                        arguments = tool_call.get("arguments", "{}")
                        if not isinstance(arguments, str):
                            # Convert dict to JSON string if needed
                            try:
                                arguments = json.dumps(arguments)
                            except (TypeError, ValueError):
                                arguments = "{}"
                                
                        formatted_tool_call = {
                            "id": tool_id,
                            "type": "function",  # Use 'function' to match OpenAI format
                            "function": {
                                "name": tool_call.get("name", ""),
                                "arguments": arguments
                            }
                        }
                        tool_calls.append(formatted_tool_call)
                    else:
                        # For any other format, try to adapt it
                        logger.warning(f"Unexpected tool call format: {tool_call}")
                        # Try to extract any useful information
                        tool_name = ""
                        tool_args = "{}"
                        
                        # Look for anything that might be a name
                        for key, value in tool_call.items():
                            if key.lower() in ["name", "function_name", "tool", "tool_name"]:
                                tool_name = str(value)
                            elif key.lower() in ["args", "arguments", "params", "parameters"]:
                                if isinstance(value, str):
                                    tool_args = value
                                else:
                                    try:
                                        tool_args = json.dumps(value)
                                    except (TypeError, ValueError):
                                        tool_args = "{}"
                        
                        formatted_tool_call = {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args
                            }
                        }
                        tool_calls.append(formatted_tool_call)
                except Exception as e:
                    logger.warning(f"Error processing tool call: {e}")
            
            # Add tool_calls to additional_kwargs
            if tool_calls:
                additional_kwargs["tool_calls"] = tool_calls
        except Exception as e:
            # Log the error but continue with the response
            logger.error(f"Error processing tool_calls: {e}")
    
    # Case 2: ASI's unique behavior - finish_reason is "tool_calls" but no tool_calls field
    # This is the key enhancement to handle ASI's specific behavior
    elif finish_reason == "tool_calls" and hasattr(self, 'tools') and self.tools:
        # ASI doesn't provide tool_calls directly, so we need to extract from context
        # Generate a tool ID
        tool_id = f"call_{str(uuid.uuid4())[:6]}"
        
        # Find the first available tool name
        tool_name = ""
        if self.tools and len(self.tools) > 0:
            # If tool_choice is specified and is a dict with a function name
            if hasattr(self, 'tool_choice'):
                if isinstance(self.tool_choice, dict) and self.tool_choice.get("type") == "function":
                    function_choice = self.tool_choice.get("function", {})
                    if isinstance(function_choice, dict) and "name" in function_choice:
                        tool_name = function_choice["name"]
                # If tool_choice is a string and not one of the special values
                elif isinstance(self.tool_choice, str) and self.tool_choice not in ["auto", "any", "none"]:
                    tool_name = self.tool_choice
            
            # Otherwise use the first tool
            if not tool_name and self.tools and len(self.tools) > 0:
                first_tool = self.tools[0]
                if isinstance(first_tool, dict) and "function" in first_tool:
                    tool_name = first_tool["function"].get("name", "")
                elif hasattr(first_tool, "name"):
                    tool_name = first_tool.name
        
        # Try to extract arguments from content
        # First, try to find JSON in the content
        arguments = "{}"
        
        # Try various methods to extract JSON from content
        json_patterns = [
            r"```json\s*(.+?)\s*```",  # JSON in code block
            r"```\s*(.+?)\s*```",       # Any code block
            r"\{.+\}",                  # Any JSON-like object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    # Test if it's valid JSON
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        arguments = match
                        break
                except json.JSONDecodeError:
                    continue
            if arguments != "{}":
                break
        
        # If we couldn't extract JSON, try to create a simple arguments object
        # based on the content
        if arguments == "{}" and tool_name:
            # For routing tools, try to extract the next agent
            if tool_name.lower() == "route":
                # Look for common patterns in the content
                next_agent = None
                reasoning = content
                info_needed = []
                
                # Try to find the next agent in the content
                agent_patterns = [
                    r"route to (?:the )?(\w+)",
                    r"send to (?:the )?(\w+)",
                    r"(\w+) agent",
                    r"(\w+) is best",
                ]
                
                for pattern in agent_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        next_agent = matches[0]
                        break
                
                # If we found a next agent, create arguments
                if next_agent:
                    args_dict = {
                        "next": next_agent,
                        "reasoning": reasoning,
                        "information_needed": info_needed
                    }
                    arguments = json.dumps(args_dict)
        
        # Create a formatted tool call
        if tool_name:
            formatted_tool_call = {
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            tool_calls.append(formatted_tool_call)
            additional_kwargs["tool_calls"] = tool_calls
    
    # Extract usage information
    usage_metadata = {}
    if "usage" in response_data:
        usage = response_data["usage"]
        usage_metadata = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    
    # Create the message
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    
    message = AIMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        usage_metadata=usage_metadata,
    )
    
    # Create the generation and result
    generation = ChatGeneration(
        message=message,
        generation_info={
            "finish_reason": finish_reason,
            "logprobs": response_data["choices"][0].get("logprobs", None),
        },
    )
    
    # Create the result with LLM output metadata
    llm_output = {"token_usage": usage_metadata, "model_name": self.model_name}
    if "system_fingerprint" in response_data:
        llm_output["system_fingerprint"] = response_data["system_fingerprint"]
            
    return ChatResult(
        generations=[generation],
        llm_output=llm_output,
    )
