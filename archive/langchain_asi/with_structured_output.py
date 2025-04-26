def with_structured_output(
    self,
    schema: Optional[Union[Dict, Type[BaseModel]]] = None,
    *,
    method: Literal["function_calling", "json_mode"] = "function_calling",
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
    """Model wrapper that returns outputs formatted to match the given schema.

    Args:
        schema:
            The output schema. Can be passed in as:
                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a TypedDict class (supported added in 0.1.9),
                - or a Pydantic class.
            If ``schema`` is a Pydantic class then the model output will be a
            Pydantic object of that class. If ``schema`` is a JSON Schema or
            OpenAI function/tool schema, then the model output will be a dict
            matching that schema.
            If ``schema`` is None, then the model output will be a dict matching
            the schema that the model generates.
        method:
            The method to use for structured output. Valid options are:
                - ``"function_calling"``: Use OpenAI function calling. This is the
                    default and recommended method for structured outputs.
                - ``"json_mode"``: Use JSON mode to generate a JSON object matching the
                    schema.
        include_raw:
            If True, the raw model response will be included in the output.
            Defaults to False.
        **kwargs:
            Additional arguments to pass to the model when generating.

    Returns:
        A Runnable that takes the same input as the model and returns a dict or
        Pydantic object matching the given schema.

    If ``include_raw`` is True, then Runnable outputs a dict with keys:
        - ``"raw"``: BaseMessage
        - ``"parsed"``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
        - ``"parsing_error"``: Optional[BaseException]
    """
    from langchain_core.output_parsers import (
        JsonOutputParser,
        PydanticOutputParser,
    )
    from langchain_core.output_parsers.openai_tools import (
        JsonOutputKeyToolsParser,
        PydanticToolsParser,
    )
    from langchain_core.runnables import (
        Runnable,
        RunnableConfig,
        RunnableMap,
        RunnablePassthrough,
    )
    from operator import itemgetter

    _ = kwargs.pop("strict", None)
    if kwargs:
        raise ValueError(f"Received unsupported arguments {kwargs}")
    is_pydantic_schema = is_basemodel_subclass(schema)
    
    if method == "function_calling":
        if schema is None:
            raise ValueError(
                "schema must be provided when method is 'function_calling'. "
                "Received None."
            )
        
        # Convert the schema to an OpenAI tool format
        formatted_tool = convert_to_openai_tool(schema)
        tool_name = formatted_tool["function"]["name"]
        
        # Create a tool that matches the schema
        from langchain_core.tools import BaseTool, tool
        
        if is_pydantic_schema:
            # For Pydantic models, use the schema directly as a tool
            @tool
            def schema_tool(**kwargs) -> dict:
                """Tool for structured output."""
                return kwargs
            
            # Set the name to match the schema
            schema_tool.name = tool_name
            
            # Bind the tool to the model
            llm = self.bind_tools([schema_tool], tool_choice="auto")
            
            # Use the standard PydanticToolsParser
            output_parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
        else:
            # For dict schemas, create a simple tool
            description = formatted_tool["function"].get("description", "Tool for structured output")
            
            # Create a simple tool
            simple_tool = BaseTool(
                name=tool_name,
                description=description,
                func=lambda **kwargs: kwargs,
            )
            
            # Bind the tool to the model
            llm = self.bind_tools([simple_tool], tool_choice="auto")
            
            # Use the standard JsonOutputKeyToolsParser
            output_parser = JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)
            
    elif method == "json_mode":
        # Use standard approach for JSON mode
        # Create a system message that instructs the model to respond with JSON
        json_system_message = (
            "Always respond with a JSON object that matches the following schema:\n"
            + (json.dumps(convert_to_json_schema(schema), indent=2) if schema else "")
        )
        
        # Bind the system message to the model
        llm = self.bind(
            stop=["\n\n"],
            system_message=json_system_message,
        )
        
        # Use the appropriate parser based on the schema type
        if is_pydantic_schema:
            output_parser = PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
        else:
            output_parser = JsonOutputParser()
    else:
        raise ValueError(
            f"method must be one of 'function_calling' or 'json_mode'. "
            f"Received {method!r}"
        )

    # Handle include_raw option
    if include_raw:
        from langchain_core.output_parsers import OutputParserException

        def parse_with_fallback(message: BaseMessage) -> Dict[str, Any]:
            try:
                parsed = output_parser.parse_result([message])
                return {
                    "raw": message,
                    "parsed": parsed,
                    "parsing_error": None,
                }
            except OutputParserException as e:
                return {"raw": message, "parsed": None, "parsing_error": e}
            except Exception as e:
                return {"raw": message, "parsed": None, "parsing_error": e}

        return llm | RunnablePassthrough.assign(parsed_with_error=parse_with_fallback)
    else:
        return llm | output_parser
