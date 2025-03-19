"""Chat models for ASI integrations."""
from __future__ import annotations

import json
import logging
import os
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Iterator, 
    AsyncIterator, Type, Sequence, Callable, Literal
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core._api import deprecated
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    generate_from_stream,
    agenerate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from pydantic import BaseModel, Field, model_validator
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    get_from_dict_or_env, 
    from_env, 
    secret_from_env, 
    get_pydantic_field_names
)
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from operator import itemgetter

import httpx
import asyncio
import requests

logger = logging.getLogger(__name__)


class ASI1ChatModel(BaseChatModel):
    """Chat model that uses ASI1's chat API.
    
    To use, you should have the environment variable ``ASI1_API_KEY`` set with your API key.
    
    Any parameters that are valid to be passed to the ASI1 Chat API can be passed in, 
    even if not explicitly saved on this class.
    
    Example:
        .. code-block:: python
            
            from langchain_asi import ASI1ChatModel
            llm = ASI1ChatModel(model_name="asi1-mini")
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default="asi1-mini")
    """Model name to use."""
    temperature: Optional[float] = 0
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""
    asi1_api_key: Optional[str] = None
    """ASI1 API key."""
    asi1_api_base: Optional[str] = Field(default="https://api.asi1.ai/v1")
    """Base URL for ASI1 API requests."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to ASI1 API."""
    streaming: bool = False
    """Whether to stream the results or not."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    stop: Optional[Union[List[str], str]] = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    http_client: Optional[Any] = None
    """HTTP client to use for API requests."""
    default_headers: Union[Dict[str, str], None] = None
    default_query: Union[Dict[str, object], None] = None
    max_retries: int = 2
    """Maximum number of retries to make when generating."""

    @model_validator(mode='after')
    def validate_environment(self) -> 'ASI1ChatModel':
        """Validate that api key exists in environment."""
        self.asi1_api_key = get_from_dict_or_env(
            {"asi1_api_key": self.asi1_api_key}, "asi1_api_key", "ASI1_API_KEY"
        )
        self.asi1_api_base = get_from_dict_or_env(
            {"asi1_api_base": self.asi1_api_base}, "asi1_api_base", "ASI1_API_BASE", "https://api.asi1.ai/v1"
        )
        
        # Validate n parameter
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")
        
        # Initialize HTTP client if not already set
        if self.http_client is None:
            self.http_client = httpx.Client(
                timeout=self.request_timeout,
                headers={
                    "Authorization": f"Bearer {self.asi1_api_key}",
                    "Content-Type": "application/json",
                    **(self.default_headers or {}),
                },
            )
        
        return self
    
    @model_validator(mode='before')
    @classmethod
    def build_extra(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = data.get("model_kwargs", {})
        for field_name in list(data):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"{field_name} is not a default parameter, "
                    f"transferring to model_kwargs."
                )
                extra[field_name] = data.pop(field_name)

        invalid_model_kwargs = set(all_required_field_names).intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        data["model_kwargs"] = extra
        return data

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling ASI1 API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "n": self.n,
            **self.model_kwargs,
        }
        
        # Add parameters if they are provided
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.stop is not None:
            params["stop"] = self.stop
            
        return params

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        """Convert a LangChain message to a dictionary for the ASI1 API."""
        message_dict = {"content": message.content}
        
        if isinstance(message, ChatMessage):
            message_dict["role"] = message.role
        elif isinstance(message, HumanMessage):
            message_dict["role"] = "user"
        elif isinstance(message, AIMessage):
            message_dict["role"] = "assistant"
            # Handle function/tool calls if present
            if message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tool_call.get("name", ""),
                            "arguments": tool_call.get("args", "{}")
                        }
                    })
                message_dict["tool_calls"] = tool_calls
                # If tool calls present, content null value should be None not empty string
                if not message_dict["content"]:
                    message_dict["content"] = None
        elif isinstance(message, SystemMessage):
            message_dict["role"] = "system"
        elif isinstance(message, FunctionMessage):
            message_dict["role"] = "function"
            message_dict["name"] = message.name
        elif isinstance(message, ToolMessage):
            message_dict["role"] = "tool"
            message_dict["tool_call_id"] = message.tool_call_id
        
        return message_dict

    def _convert_dict_to_message(self, message_dict: Dict[str, Any]) -> BaseMessage:
        """Convert a dictionary to a LangChain message."""
        role = message_dict.get("role")
        content = message_dict.get("content", "")
        
        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            # Handle function/tool calls if present
            additional_kwargs = {}
            tool_calls = []
            invalid_tool_calls = []
            
            if raw_tool_calls := message_dict.get("tool_calls"):
                additional_kwargs["tool_calls"] = raw_tool_calls
                for raw_tool_call in raw_tool_calls:
                    try:
                        tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                    except Exception as e:
                        invalid_tool_calls.append(
                            make_invalid_tool_call(raw_tool_call, str(e))
                        )
            
            return AIMessage(
                content=content or "",
                additional_kwargs=additional_kwargs,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
            )
        elif role == "system":
            return SystemMessage(content=content)
        elif role == "function":
            return FunctionMessage(
                content=content, 
                name=message_dict.get("name", "")
            )
        elif role == "tool":
            return ToolMessage(
                content=content,
                tool_call_id=message_dict.get("tool_call_id", "")
            )
        else:
            return ChatMessage(content=content, role=role)

    def _create_chat_result(self, response: Dict[str, Any]) -> ChatResult:
        """Create a ChatResult from the response."""
        generations = []
        
        for choice in response.get("choices", []):
            message_dict = choice.get("message", {})
            message = self._convert_dict_to_message(message_dict)
            
            generation_info = {
                "finish_reason": choice.get("finish_reason"),
            }
            
            generation = ChatGeneration(
                message=message,
                generation_info=generation_info
            )
            generations.append(generation)
        
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": response.get("model", self.model_name),
        }
        
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completions."""
        if self.streaming:
            # If streaming is enabled, use _stream and collect the chunks
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        
        # Prepare the request payload
        params = self._default_params.copy()
        if stop:
            params["stop"] = stop
        params.update(kwargs)
        
        # Convert messages to the format expected by the ASI1 API
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        payload = {
            **params,
            "messages": message_dicts,
        }
        
        # Make the API request
        try:
            response = requests.post(
                f"{self.asi1_api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.asi1_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            return self._create_chat_result(response.json())
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error calling ASI1 API: {e}")

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions."""
        # Prepare the request payload
        params = self._default_params.copy()
        params["stream"] = True  # Ensure streaming is enabled
        if stop:
            params["stop"] = stop
        params.update(kwargs)
        
        # Convert messages to the format expected by the ASI1 API
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        payload = {
            **params,
            "messages": message_dicts,
        }
        
        # Make the streaming API request
        try:
            response = requests.post(
                f"{self.asi1_api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.asi1_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.request_timeout,
                stream=True,
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # Remove 'data: ' prefix if present
                if line.startswith(b"data: "):
                    line = line[6:]
                    
                if line.strip() == b"[DONE]":
                    break
                    
                try:
                    chunk = json.loads(line)
                    
                    # Extract the delta content
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                        
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    role = delta.get("role")
                    
                    # Create the appropriate message chunk
                    if role == "assistant" or not role:
                        message_chunk = AIMessageChunk(content=content)
                    elif role == "user":
                        message_chunk = HumanMessageChunk(content=content)
                    elif role == "system":
                        message_chunk = SystemMessageChunk(content=content)
                    else:
                        message_chunk = ChatMessageChunk(content=content, role=role)
                    
                    # Create the generation chunk
                    generation_info = {
                        "finish_reason": choice.get("finish_reason"),
                    }
                    generation_chunk = ChatGenerationChunk(
                        message=message_chunk,
                        generation_info=generation_info if generation_info["finish_reason"] else None
                    )
                    
                    # Send the chunk to the run manager if provided
                    if run_manager:
                        run_manager.on_llm_new_token(
                            token=content,
                            chunk=generation_chunk,
                        )
                        
                    yield generation_chunk
                except json.JSONDecodeError:
                    continue
                    
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error calling ASI1 API: {e}")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate chat completions."""
        if self.streaming:
            # If streaming is enabled, use _astream and collect the chunks
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        
        # Prepare the request payload
        params = self._default_params.copy()
        if stop:
            params["stop"] = stop
        params.update(kwargs)
        
        # Convert messages to the format expected by the ASI1 API
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        payload = {
            **params,
            "messages": message_dicts,
        }
        
        # Make the API request asynchronously
        try:
            # Initialize async client if not already set
            if self.async_client is None:
                self.async_client = httpx.AsyncClient(
                    timeout=self.request_timeout,
                    headers={
                        "Authorization": f"Bearer {self.asi1_api_key}",
                        "Content-Type": "application/json",
                        **(self.default_headers or {}),
                    },
                )
            
            response = await self.async_client.post(
                f"{self.asi1_api_base}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            return self._create_chat_result(response.json())
        except httpx.HTTPError as e:
            raise ValueError(f"Error calling ASI1 API: {e}")

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously stream chat completions."""
        # Prepare the request payload
        params = self._default_params.copy()
        params["stream"] = True  # Ensure streaming is enabled
        if stop:
            params["stop"] = stop
        params.update(kwargs)
        
        # Convert messages to the format expected by the ASI1 API
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        payload = {
            **params,
            "messages": message_dicts,
        }
        
        # Initialize async client if not already set
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(
                timeout=self.request_timeout,
                headers={
                    "Authorization": f"Bearer {self.asi1_api_key}",
                    "Content-Type": "application/json",
                    **(self.default_headers or {}),
                },
            )
        
        # Make the streaming API request asynchronously
        try:
            async with self.async_client.stream(
                "POST",
                f"{self.asi1_api_base}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                        
                    # Remove 'data: ' prefix if present
                    if line.startswith("data: "):
                        line = line[6:]
                        
                    if line.strip() == "[DONE]":
                        break
                        
                    try:
                        chunk = json.loads(line)
                        
                        # Extract the delta content
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                            
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        role = delta.get("role")
                        
                        # Create the appropriate message chunk
                        if role == "assistant" or not role:
                            message_chunk = AIMessageChunk(content=content)
                        elif role == "user":
                            message_chunk = HumanMessageChunk(content=content)
                        elif role == "system":
                            message_chunk = SystemMessageChunk(content=content)
                        else:
                            message_chunk = ChatMessageChunk(content=content, role=role)
                        
                        # Create the generation chunk
                        generation_info = {
                            "finish_reason": choice.get("finish_reason"),
                        }
                        generation_chunk = ChatGenerationChunk(
                            message=message_chunk,
                            generation_info=generation_info if generation_info["finish_reason"] else None
                        )
                        
                        # Send the chunk to the run manager if provided
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                token=content,
                                chunk=generation_chunk,
                            )
                            
                        yield generation_chunk
                    except json.JSONDecodeError:
                        continue
                        
        except httpx.HTTPError as e:
            raise ValueError(f"Error calling ASI1 API: {e}")

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Get the request payload for the ASI1 API."""
        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop

        payload = {**self._default_params, **kwargs}
        payload["messages"] = [self._convert_message_to_dict(m) for m in messages]
        return payload

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        """Combine LLM outputs."""
        overall_token_usage = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output.get("token_usage")
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "asi1"

    def bind_tools(
        self, tools: Sequence[Union[Dict[str, Any], Type[BaseModel], BaseTool]]
    ) -> Runnable:
        """Bind tools to the model.
        
        Args:
            tools: A sequence of tools to bind to the model. Can be either dictionaries,
                Pydantic models, or LangChain tools.
                
        Returns:
            A runnable that will use the tools.
        """
        # Convert tools to the format expected by ASI1 API
        converted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                converted_tools.append(tool)
            elif _is_pydantic_class(tool):
                converted_tools.append(convert_to_openai_tool(tool))
            elif isinstance(tool, BaseTool):
                converted_tools.append(convert_to_openai_tool(tool))
            else:
                raise ValueError(
                    f"Tool {tool} is not a dictionary, pydantic model, or langchain tool."
                )
        
        # Create a new model with the tools
        model_kwargs = self.model_kwargs.copy()
        model_kwargs["tools"] = converted_tools
        
        return self.bind(tools=converted_tools)

    def with_structured_output(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        *,
        include_raw: bool = False,
        name: Optional[str] = None,
    ) -> Runnable:
        """Use schema to create a structured output.
        
        Args:
            schema: Either a pydantic model or a dictionary representing a JSON schema.
            include_raw: Whether to include the raw response from the model in the output.
            name: Name of the function to use for the structured output.
            
        Returns:
            A runnable that will return a structured output.
        """
        if is_basemodel_subclass(schema):
            # If the schema is a pydantic model, use the PydanticOutputParser
            output_parser: OutputParserLike = PydanticOutputParser(pydantic_object=schema)
            # For Pydantic models, we need to create the function schema manually if name is provided
            if name:
                function_name = name
                openai_schema = convert_to_openai_function(schema)
                openai_schema["name"] = function_name
            else:
                openai_schema = convert_to_openai_function(schema)
        else:
            # If the schema is a dictionary, use the JsonOutputParser
            output_parser = JsonOutputParser()
            openai_schema = {
                "name": name or "output_formatter",
                "description": "Output formatter. Should call this function.",
                "parameters": schema,
            }
        
        if include_raw:
            # If include_raw is True, return both the raw response and the structured output
            outputs = {"raw": RunnablePassthrough(), "parsed": output_parser}
            return self.bind(
                functions=[openai_schema], function_call={"name": openai_schema["name"]}
            ).pipe(outputs)
        else:
            # Otherwise, just return the structured output
            return self.bind(
                functions=[openai_schema], function_call={"name": openai_schema["name"]}
            ).pipe(output_parser)


def _is_pydantic_class(obj: Any) -> bool:
    """Check if an object is a Pydantic class."""
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _lc_tool_call_to_asi_tool_call(tool_call: ToolCall) -> Dict[str, Any]:
    """Convert a LangChain tool call to an ASI tool call."""
    return {
        "id": tool_call.get("id", ""),
        "type": "function",
        "function": {
            "name": tool_call.get("name", ""),
            "arguments": tool_call.get("args", "{}")
        }
    }


def _lc_invalid_tool_call_to_asi_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> Dict[str, Any]:
    """Convert a LangChain invalid tool call to an ASI tool call."""
    return {
        "id": invalid_tool_call.id,
        "type": "function",
        "function": {
            "name": invalid_tool_call.name,
            "arguments": invalid_tool_call.args
        }
    }