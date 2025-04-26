"""ASI Chat wrapper."""

from __future__ import annotations

import json
import logging
import os
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs

logger = logging.getLogger(__name__)

# Define retry decorator functions
def _create_retry_decorator(
    llm: Any, run_manager: Optional[CallbackManagerForLLMRun] = None
) -> Callable[[Any], Any]:
    """Create a retry decorator for the LLM."""
    def _backoff(attempt: int) -> float:
        # Exponential backoff with jitter
        import random
        return 2 ** attempt + random.random()

    decorator = create_base_retry_decorator(
        error_types=(
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            json.JSONDecodeError,
        ),
        max_retries=llm.max_retries,
        run_manager=run_manager,
        backoff_function=_backoff,
    )
    return decorator


def _create_async_retry_decorator(
    llm: Any, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None
) -> Callable[[Any], Any]:
    """Create an async retry decorator for the LLM."""
    def _backoff(attempt: int) -> float:
        # Exponential backoff with jitter
        import random
        return 2 ** attempt + random.random()

    decorator = create_base_retry_decorator(
        error_types=(
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            json.JSONDecodeError,
        ),
        max_retries=llm.max_retries,
        run_manager=run_manager,
        backoff_function=_backoff,
    )
    return decorator

# Utility functions
def _lc_tool_call_to_asi_tool_call(tool_call: ToolCall) -> Dict[str, Any]:
    """Convert a LangChain tool call to an ASI tool call."""
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": tool_call.args,
        },
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
            "arguments": invalid_tool_call.args,
        },
    }


class _FunctionCall(BaseModel):
    """Function call model."""

    name: str


class ChatASI(BaseChatModel):
    """ASI Chat large language models API.

    To use, you should have the environment variable ``ASI_API_KEY`` set with your API key,
    or pass it as a constructor parameter.

    Example:
        .. code-block:: python

            from langchain_asi import ChatASI
            chat = ChatASI(model_name="asi1-mini")
    """

    client: httpx.Client = Field(default=None, exclude=True)  # type: ignore
    async_client: httpx.AsyncClient = Field(default=None, exclude=True)  # type: ignore
    model_name: str = "asi1-mini"
    temperature: float = 0.7
    top_p: float = 1.0
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    asi_api_key: Optional[str] = None
    asi_api_base: Optional[str] = None
    request_timeout: float = 60.0
    max_retries: int = 6
    streaming: bool = False
    max_tokens: Optional[int] = None
    n: int = 1
    verbose: bool = False
    http_client: Optional[httpx.Client] = None
    http_async_client: Optional[httpx.AsyncClient] = None
    asi_organization: Optional[str] = None

    model_config = Field(default_factory=dict)

    @root_validator
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        # Get API key from parameters or environment variable
        values["asi_api_key"] = values["asi_api_key"] or os.environ.get("ASI_API_KEY")
        if values["asi_api_key"] is None:
            raise ValueError(
                "ASI API key not provided. You can pass it as asi_api_key or set it as "
                "an environment variable ASI_API_KEY."
            )

        # Set API base URL based on model name, environment variable, or explicit setting
        if values["asi_api_base"] is None:
            # If environment variable is set, use that as the base URL
            env_api_base = os.environ.get("ASI_API_BASE")
            if env_api_base:
                values["asi_api_base"] = env_api_base
            # If model_name starts with 'asi1-', use ASI1 endpoint
            elif values["model_name"] and values["model_name"].startswith("asi1-"):
                values["asi_api_base"] = "https://api.asi1.ai/v1"
            # Default to standard ASI endpoint
            else:
                values["asi_api_base"] = "https://api.asi.ai/v1"
        
        # Log the selected API base URL if verbose
        if values.get("verbose", False):
            logger.info(f"Using ASI API base URL: {values['asi_api_base']}")
            logger.info(f"Using ASI model: {values['model_name']}")

        # Initialize httpx clients with proper headers and base URL
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {values['asi_api_key']}",
        }

        # Initialize the client if not already provided
        if values.get("http_client") is not None:
            values["client"] = values["http_client"]
        elif values.get("client") is None:
            values["client"] = httpx.Client(
                base_url=values["asi_api_base"],
                headers=headers,
                timeout=values.get("request_timeout", 60.0),
            )

        # Initialize the async client if not already provided
        if values.get("http_async_client") is not None:
            values["async_client"] = values["http_async_client"]
        elif values.get("async_client") is None:
            values["async_client"] = httpx.AsyncClient(
                base_url=values["asi_api_base"],
                headers=headers,
                timeout=values.get("request_timeout", 60.0),
            )

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "asi-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Return the secrets to use for serialization."""
        return {"asi_api_key": "ASI_API_KEY"}
    
    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "asi"]

    @property
    def supported_usage_metadata_details(self) -> Dict[
        Literal["invoke", "stream"],
        List[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        """(dict) what usage metadata details are emitted in invoke and stream. Only
        needs to be overridden if these details are returned by the model."""
        return {"invoke": [], "stream": []}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            **{"temperature": self.temperature},
            **{"top_p": self.top_p},
            **{"max_tokens": self.max_tokens},
            **self.model_kwargs,
        }

    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling ASI API."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "stream": self.streaming,
            **self.model_kwargs,
        }

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": stop,
        }
        params.update(self.model_kwargs)
        params.update(kwargs)
        return params

    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for the API request."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.asi_api_key}",
        }
        return headers

    def _create_message_dicts(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Create message dicts for the API request."""
        message_dicts = []
        system_messages = []

        # First, collect all system messages
        for message in messages:
            if isinstance(message, SystemMessage):
                system_messages.append(message.content)
            else:
                # For non-system messages, convert to the appropriate format
                if isinstance(message, HumanMessage):
                    message_dict = {"role": "user", "content": message.content}
                elif isinstance(message, AIMessage):
                    message_dict = {"role": "assistant", "content": message.content}
                    # Add tool calls if present
                    if message.tool_calls:
                        tool_calls = [
                            _lc_tool_call_to_asi_tool_call(tool_call)
                            for tool_call in message.tool_calls
                        ]
                        message_dict["tool_calls"] = tool_calls
                elif isinstance(message, FunctionMessage):
                    message_dict = {
                        "role": "function",
                        "name": message.name,
                        "content": message.content,
                    }
                elif isinstance(message, ToolMessage):
                    message_dict = {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                else:
                    raise ValueError(f"Unknown message type: {type(message)}")

                message_dicts.append(message_dict)

        # If there are system messages, combine them and add as the first message
        if system_messages:
            combined_system_message = " ".join(system_messages)
            message_dicts.insert(0, {"role": "system", "content": combined_system_message})

        return message_dicts

    def _create_chat_result(self, response_data: Dict[str, Any]) -> ChatResult:
        """Create a ChatResult from the response data.

        Args:
            response_data: The response data from the API.

        Returns:
            A ChatResult containing the generated responses.
        """
        # Extract the response message
        if "choices" not in response_data:
            raise ValueError(f"Invalid response format: {response_data}")

        message_data = response_data["choices"][0]["message"]
        content = message_data.get("content", "")

        # Extract additional information if available
        additional_kwargs = {}
        if "function_call" in message_data:
            additional_kwargs["function_call"] = message_data["function_call"]
        if "tool_calls" in message_data:
            additional_kwargs["tool_calls"] = message_data["tool_calls"]
        if "thought" in message_data:
            additional_kwargs["thought"] = message_data["thought"]

        # Extract usage information if available
        usage_metadata = None
        if "usage" in response_data:
            usage = response_data["usage"]
            usage_metadata = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        # Create the AI message
        message = AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,
        )

        # Create the generation and result
        generation = ChatGeneration(
            message=message,
            generation_info={
                "finish_reason": response_data["choices"][0].get("finish_reason", None),
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

    def _process_chunk(self, chunk_data: Dict[str, Any]) -> Optional[ChatGenerationChunk]:
        """Process a chunk from the streaming response.

        Args:
            chunk_data: The chunk data from the streaming response.

        Returns:
            A ChatGenerationChunk if the chunk contains a delta, None otherwise.
        """
        if "choices" not in chunk_data or not chunk_data["choices"]:
            return None

        choice = chunk_data["choices"][0]
        if "delta" not in choice:
            return None

        delta = choice["delta"]
        content = delta.get("content", "")
        role = delta.get("role")

        # Extract additional information if available
        additional_kwargs = {}
        if "function_call" in delta:
            additional_kwargs["function_call"] = delta["function_call"]
        if "tool_calls" in delta:
            additional_kwargs["tool_calls"] = delta["tool_calls"]
        if "thought" in delta:
            additional_kwargs["thought"] = delta["thought"]

        # Extract usage information if available
        usage_metadata = None
        if "usage" in chunk_data:
            usage = chunk_data["usage"]
            usage_metadata = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        # Create the message chunk
        message_chunk = AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,
        )

        # Create the generation chunk
        generation_info = {}
        if "finish_reason" in choice and choice["finish_reason"] is not None:
            generation_info["finish_reason"] = choice["finish_reason"]
        if "logprobs" in choice and choice["logprobs"] is not None:
            generation_info["logprobs"] = choice["logprobs"]

        return ChatGenerationChunk(
            message=message_chunk,
            generation_info=generation_info if generation_info else None,
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the chat response in chunks.

        Args:
            messages: The messages to send to the ASI API.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for tracking the run.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            An iterator of ChatGenerationChunk objects containing the generated text.
        """
        message_dicts = self._create_message_dicts(messages)
        params = self._default_params()
        params.update(kwargs)

        if stop:
            params["stop"] = stop

        # Prepare the data for the API request
        data = {
            "model": self.model_name,
            "messages": message_dicts,
            "stream": True,
            **params,
        }

        # Log the request if verbose
        if self.verbose:
            print(f"Request to ASI API: {self.asi_api_base}/v1/chat/completions")
            print(f"Request data: {data}")

        # Create headers for the request
        headers = {
            "Authorization": f"Bearer {self.asi_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Add organization header if provided
        if hasattr(self, "asi_organization") and self.asi_organization:
            headers["ASI-Organization"] = self.asi_organization

        # Create a retry decorator
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _stream_with_retry() -> Iterator[ChatGenerationChunk]:
            url = f"{self.asi_api_base}/v1/chat/completions"
            
            with self.client.stream(
                "POST",
                url,
                headers=headers,
                json=data,
                timeout=self.request_timeout,
            ) as response:
                self._raise_on_error(response)
                
                # Initialize variables to track the message being built
                default_chunk_class = AIMessageChunk
                
                # Process each line in the response
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    line_str = line.decode("utf-8")
                    
                    # Handle SSE format
                    if line_str.startswith("data: "):
                        line_str = line_str[6:]
                        
                    if line_str.strip() == "[DONE]":
                        break
                        
                    try:
                        chunk_data = json.loads(line_str)
                        
                        # Process each choice in the chunk
                        if "choices" in chunk_data:
                            for choice in chunk_data["choices"]:
                                delta = choice.get("delta", {})
                                
                                # Get content from delta
                                content = delta.get("content", "")
                                
                                # Get role from delta if present
                                role = delta.get("role")
                                
                                # Handle tool calls if present
                                tool_calls = delta.get("tool_calls", [])
                                
                                # Create generation info if finish reason is present
                                generation_info = None
                                if finish_reason := choice.get("finish_reason"):
                                    generation_info = {"finish_reason": finish_reason}
                                
                                # Create and yield the chunk
                                message_chunk = AIMessageChunk(
                                    content=content,
                                    role=role,
                                    tool_calls=tool_calls if tool_calls else None,
                                )
                                
                                chunk = ChatGenerationChunk(
                                    message=message_chunk,
                                    generation_info=generation_info,
                                )
                                
                                # Update default chunk class for future chunks
                                default_chunk_class = message_chunk.__class__
                                
                                # Notify run manager if present
                                if run_manager:
                                    run_manager.on_llm_new_token(
                                        token=content, chunk=chunk
                                    )
                                    
                                yield chunk
                    except json.JSONDecodeError as e:
                        if self.verbose:
                            print(f"Error decoding JSON: {e}")
                            print(f"Line: {line_str}")
                        continue

        return _stream_with_retry()

    def _process_chat_response_stream(
        self, response: httpx.Response
    ) -> Iterator[ChatGenerationChunk]:
        """Process the streaming response from the API.

        Args:
            response: The streaming response from the API.

        Returns:
            An iterator of ChatGenerationChunk objects containing the generated text.
        """
        # Initialize variables to track the message being built
        default_chunk_class = AIMessageChunk
        
        # Process each line in the response
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode("utf-8")
            
            # Handle SSE format
            if line_str.startswith("data: "):
                line_str = line_str[6:]
                
            if line_str.strip() == "[DONE]":
                break
                
            try:
                chunk_data = json.loads(line_str)
                
                # Process each choice in the chunk
                if "choices" in chunk_data:
                    for choice in chunk_data["choices"]:
                        delta = choice.get("delta", {})
                        
                        # Get content from delta
                        content = delta.get("content", "")
                        
                        # Get role from delta if present
                        role = delta.get("role")
                        
                        # Handle tool calls if present
                        tool_calls = delta.get("tool_calls", [])
                        
                        # Create generation info if finish reason is present
                        generation_info = None
                        if finish_reason := choice.get("finish_reason"):
                            generation_info = {"finish_reason": finish_reason}
                        
                        # Create and yield the chunk
                        message_chunk = AIMessageChunk(
                            content=content,
                            role=role,
                            tool_calls=tool_calls if tool_calls else None,
                        )
                        
                        chunk = ChatGenerationChunk(
                            message=message_chunk,
                            generation_info=generation_info,
                        )
                        
                        # Update default chunk class for future chunks
                        default_chunk_class = message_chunk.__class__
                        
                        yield chunk
            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"Error decoding JSON: {e}")
                    print(f"Line: {line_str}")
                continue

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the chat response in chunks asynchronously.

        Args:
            messages: The messages to send to the ASI API.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for tracking the run.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            An async iterator of ChatGenerationChunk objects containing the generated text.
        """
        message_dicts = self._create_message_dicts(messages)
        params = self._default_params()
        params.update(kwargs)

        if stop:
            params["stop"] = stop

        # Prepare the data for the API request
        data = {
            "model": self.model_name,
            "messages": message_dicts,
            "stream": True,
            **params,
        }

        # Log the request if verbose
        if self.verbose:
            print(f"Request to ASI API: {self.asi_api_base}/v1/chat/completions")
            print(f"Request data: {data}")

        # Create headers for the request
        headers = {
            "Authorization": f"Bearer {self.asi_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Add organization header if provided
        if hasattr(self, "asi_organization") and self.asi_organization:
            headers["ASI-Organization"] = self.asi_organization

        # Create a retry decorator
        retry_decorator = _create_async_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _astream_with_retry() -> AsyncIterator[ChatGenerationChunk]:
            url = f"{self.asi_api_base}/v1/chat/completions"
            
            async with self.async_client.stream(
                "POST",
                url,
                headers=headers,
                json=data,
                timeout=self.request_timeout,
            ) as response:
                await self._araise_on_error(response)
                
                # Initialize variables to track the message being built
                default_chunk_class = AIMessageChunk
                
                # Process each line in the response
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    line_str = line.decode("utf-8")
                    
                    # Handle SSE format
                    if line_str.startswith("data: "):
                        line_str = line_str[6:]
                        
                    if line_str.strip() == "[DONE]":
                        break
                        
                    try:
                        chunk_data = json.loads(line_str)
                        
                        # Process each choice in the chunk
                        if "choices" in chunk_data:
                            for choice in chunk_data["choices"]:
                                delta = choice.get("delta", {})
                                
                                # Get content from delta
                                content = delta.get("content", "")
                                
                                # Get role from delta if present
                                role = delta.get("role")
                                
                                # Handle tool calls if present
                                tool_calls = delta.get("tool_calls", [])
                                
                                # Create generation info if finish reason is present
                                generation_info = None
                                if finish_reason := choice.get("finish_reason"):
                                    generation_info = {"finish_reason": finish_reason}
                                
                                # Create and yield the chunk
                                message_chunk = AIMessageChunk(
                                    content=content,
                                    role=role,
                                    tool_calls=tool_calls if tool_calls else None,
                                )
                                
                                chunk = ChatGenerationChunk(
                                    message=message_chunk,
                                    generation_info=generation_info,
                                )
                                
                                # Update default chunk class for future chunks
                                default_chunk_class = message_chunk.__class__
                                
                                # Notify run manager if present
                                if run_manager:
                                    await run_manager.on_llm_new_token(
                                        token=content, chunk=chunk
                                    )
                                    
                                yield chunk
                    except json.JSONDecodeError as e:
                        if self.verbose:
                            print(f"Error decoding JSON: {e}")
                            print(f"Line: {line_str}")
                        continue

        return await _astream_with_retry()

    async def _aprocess_chat_response_stream(
        self, response: httpx.AsyncResponse
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Process the streaming response from the API asynchronously.

        Args:
            response: The streaming response from the API.

        Returns:
            An async iterator of ChatGenerationChunk objects containing the generated text.
        """
        # Initialize variables to track the message being built
        default_chunk_class = AIMessageChunk
        
        # Process each line in the response
        async for line in response.aiter_lines():
            if not line:
                continue
            
            line_str = line.decode("utf-8")
            
            # Handle SSE format
            if line_str.startswith("data: "):
                line_str = line_str[6:]
                
            if line_str.strip() == "[DONE]":
                break
                
            try:
                chunk_data = json.loads(line_str)
                
                # Process each choice in the chunk
                if "choices" in chunk_data:
                    for choice in chunk_data["choices"]:
                        delta = choice.get("delta", {})
                        
                        # Get content from delta
                        content = delta.get("content", "")
                        
                        # Get role from delta if present
                        role = delta.get("role")
                        
                        # Handle tool calls if present
                        tool_calls = delta.get("tool_calls", [])
                        
                        # Create generation info if finish reason is present
                        generation_info = None
                        if finish_reason := choice.get("finish_reason"):
                            generation_info = {"finish_reason": finish_reason}
                        
                        # Create and yield the chunk
                        message_chunk = AIMessageChunk(
                            content=content,
                            role=role,
                            tool_calls=tool_calls if tool_calls else None,
                        )
                        
                        chunk = ChatGenerationChunk(
                            message=message_chunk,
                            generation_info=generation_info,
                        )
                        
                        # Update default chunk class for future chunks
                        default_chunk_class = message_chunk.__class__
                        
                        yield chunk
            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"Error decoding JSON: {e}")
                    print(f"Line: {line_str}")
                continue

    def _raise_on_error(self, response: httpx.Response) -> None:
        """Raise an error if the response is an error."""
        if httpx.codes.is_error(response.status_code):
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
            except Exception:
                error_message = response.text
            
            raise httpx.HTTPStatusError(
                f"Error response {response.status_code} "
                f"while fetching {response.url}: {error_message}",
                request=response.request,
                response=response,
            )

    async def _araise_on_error(self, response: httpx.Response) -> None:
        """Raise an error if the response is an error (async version)."""
        if httpx.codes.is_error(response.status_code):
            try:
                error_data = await response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
            except Exception:
                error_message = await response.aread()
                error_message = error_message.decode("utf-8")
            
            raise httpx.HTTPStatusError(
                f"Error response {response.status_code} "
                f"while fetching {response.url}: {error_message}",
                request=response.request,
                response=response,
            )

    async def acompletion_with_retry(
        self,
        messages: List[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use tenacity to retry the async completion call."""
        retry_decorator = _create_async_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry() -> Any:
            # Construct the URL for chat completions
            url = f"{self.asi_api_base}/v1/chat/completions"
            
            # Create headers for the request
            headers = {
                "Authorization": f"Bearer {self.asi_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Add organization header if provided
            if hasattr(self, "asi_organization") and self.asi_organization:
                headers["ASI-Organization"] = self.asi_organization
            
            response = await self.async_client.post(
                url,
                headers=headers,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    **kwargs,
                },
                timeout=self.request_timeout
            )
            await self._araise_on_error(response)
            return response.json()

        return await _completion_with_retry()


    def completion_with_retry(
        self,
        messages: List[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry() -> Any:
            # Construct the URL for chat completions
            url = f"{self.asi_api_base}/v1/chat/completions"
            
            # Create headers for the request
            headers = {
                "Authorization": f"Bearer {self.asi_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Add organization header if provided
            if hasattr(self, "asi_organization") and self.asi_organization:
                headers["ASI-Organization"] = self.asi_organization
            
            response = self.client.post(
                url,
                headers=headers,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    **kwargs,
                },
                timeout=self.request_timeout
            )
            self._raise_on_error(response)
            return response.json()

        return _completion_with_retry()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call the ASI API to generate a chat completion.

        Args:
            messages: The list of messages to generate a response for.
            stop: The list of stop sequences to use when generating.
            run_manager: The callback manager to use for this run.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            The generated ChatResult.

        Raises:
            ValueError: If the response from the API is not as expected.
        """
        if self.streaming:
            return generate_from_stream(
                self._stream(
                    messages=messages, stop=stop, run_manager=run_manager, **kwargs
                )
            )

        message_dicts = self._create_message_dicts(messages)
        params = self._default_params()
        params.update(kwargs)

        if stop:
            params["stop"] = stop

        # Prepare the data for the API request
        data = {
            "model": self.model_name,
            "messages": message_dicts,
            **params,
        }

        # Make the API request with retry capability
        response_data = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **kwargs
        )

        return self._process_chat_response(response_data)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously call the ASI API to generate a chat completion.

        Args:
            messages: The list of messages to generate a response for.
            stop: The list of stop sequences to use when generating.
            run_manager: The callback manager to use for this run.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            The generated ChatResult.

        Raises:
            ValueError: If the response from the API is not as expected.
        """
        if self.streaming:
            return await agenerate_from_stream(
                self._astream(
                    messages=messages, stop=stop, run_manager=run_manager, **kwargs
                )
            )

        message_dicts = self._create_message_dicts(messages)
        params = self._default_params()
        params.update(kwargs)

        if stop:
            params["stop"] = stop

        # Prepare the data for the API request
        data = {
            "model": self.model_name,
            "messages": message_dicts,
            **params,
        }

        # Make the API request with retry capability
        response_data = await self.acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **kwargs
        )

        return self._process_chat_response(response_data)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
            tool_choice: Which tool to require the model to call.
            **kwargs: Additional parameters to pass to the Runnable constructor.

        Returns:
            A Runnable that takes the same inputs as the chat model and returns a BaseMessage.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "none", "required")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Args:
            text: The text to count tokens for.

        Returns:
            The number of tokens in the text.
        """
        # ASI doesn't provide a tokenizer, so we use a simple approximation
        # This is a very rough estimate and should be replaced with a proper tokenizer
        # when available from ASI
        return len(text.split())

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in the messages.

        Args:
            messages: The messages to count tokens for.

        Returns:
            The number of tokens in the messages.
        """
        num_tokens = 0
        for message in messages:
            num_tokens += self.get_num_tokens(message.content or "")
            # Add tokens for each tool call if present
            for tool_call in message.tool_calls:
                # Add tokens for function name and arguments
                num_tokens += self.get_num_tokens(tool_call.get("name", ""))
                num_tokens += self.get_num_tokens(json.dumps(tool_call.get("args", {})))
            # Add tokens for additional kwargs if present
            if hasattr(message, "additional_kwargs") and message.additional_kwargs:
                num_tokens += self.get_num_tokens(json.dumps(message.additional_kwargs))
        return num_tokens

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:
                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a TypedDict class (supported added in 0.1.9),
                - or a Pydantic class.
                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.
            method: The method for steering model generation, either "function_calling"
                or "json_mode". If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If "json_mode" then ASI's JSON mode will be
                used. Note that if using "json_mode" then you must include instructions
                for formatting the output into the desired schema into the model call.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes same inputs as a BaseChatModel.

            If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs
            an instance of ``schema`` (i.e., a Pydantic object).

            Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            If ``include_raw`` is True, then Runnable outputs a dict with keys:
                - ``"raw"``: BaseMessage
                - ``"parsed"``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
                - ``"parsing_error"``: Optional[BaseException]
        """
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be provided when method='function_calling'"
                )

            if is_basemodel_subclass(schema):
                parser: OutputParserLike = PydanticToolsParser(schema)
                schema_dict = convert_to_openai_tool(schema)
            else:
                schema_dict = schema
                parser = JsonOutputKeyToolsParser(key=schema_dict["function"]["name"])

            if include_raw:
                # Return both the raw message and the parsed output
                def _parse_with_error(
                    message: BaseMessage,
                ) -> Dict[str, Any]:
                    try:
                        parsed = parser.parse_tool_call(message)
                        return {
                            "raw": message,
                            "parsed": parsed,
                            "parsing_error": None,
                        }
                    except Exception as e:
                        return {
                            "raw": message,
                            "parsed": None,
                            "parsing_error": e,
                        }

                chain = self.bind_tools(
                    tools=[schema_dict], tool_choice=schema_dict["function"]["name"]
                )
                chain = chain.with_config({"run": {"callbacks": kwargs.get("callbacks")}})
                return chain.bind(lambda x: _parse_with_error(x))
            else:
                # Return only the parsed output
                chain = self.bind_tools(
                    tools=[schema_dict], tool_choice=schema_dict["function"]["name"]
                )
                chain = chain.with_config({"run": {"callbacks": kwargs.get("callbacks")}})
                return chain.bind(lambda x: parser.parse_tool_call(x))
        elif method == "json_mode":
            if schema is None:
                # If no schema is provided, just parse as JSON
                parser = JsonOutputParser()
            elif is_basemodel_subclass(schema):
                # If schema is a Pydantic class, use PydanticOutputParser
                parser = PydanticOutputParser(pydantic_object=schema)
            else:
                # If schema is a dict, use JsonOutputParser
                parser = JsonOutputParser()

            if include_raw:
                # Return both the raw message and the parsed output
                def _parse_with_error(
                    message: BaseMessage,
                ) -> Dict[str, Any]:
                    try:
                        parsed = parser.parse(message.content)
                        return {
                            "raw": message,
                            "parsed": parsed,
                            "parsing_error": None,
                        }
                    except Exception as e:
                        return {
                            "raw": message,
                            "parsed": None,
                            "parsing_error": e,
                        }

                return (
                    self.with_config(
                        {"model_kwargs": {"response_format": {"type": "json_object"}}}
                    )
                    .with_config({"run": {"callbacks": kwargs.get("callbacks")}})
                    .bind(lambda x: _parse_with_error(x))
                )
            else:
                # Return only the parsed output
                return (
                    self.with_config(
                        {"model_kwargs": {"response_format": {"type": "json_object"}}}
                    )
                    .with_config({"run": {"callbacks": kwargs.get("callbacks")}})
                    .bind(lambda x: parser.parse(x.content))
                )
        else:
            raise ValueError(
                f"method must be 'function_calling' or 'json_mode', got {method}"
            )

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        """Stream the chat response in chunks.

        Args:
            input: The input messages to send to the ASI API.
            config: A RunnableConfig to use for the stream.
            stop: A list of strings to stop generation when encountered.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            An iterator of BaseMessageChunk objects containing the generated text.
        """
        messages = self._convert_input_to_messages(input)
        message_dicts = self._create_message_dicts(messages)
        params = self._default_params()
        params.update(kwargs)

        if stop:
            params["stop"] = stop

        # Prepare the data for the API request
        data = {
            "model": self.model_name,
            "messages": message_dicts,
            "stream": True,
            **params,
        }

        # Log the request if verbose
        if self.verbose:
            print(f"Request to ASI API: {self.asi_api_base}/v1/chat/completions")
            print(f"Request data: {data}")

        # Create headers for the request
        headers = self._create_headers()

        url = f"{self.asi_api_base}/v1/chat/completions"
        
        with self.client.stream(
            "POST",
            url,
            headers=headers,
            json=data,
            timeout=self.request_timeout,
        ) as response:
            self._raise_on_error(response)
            
            # Initialize variables to track the message being built
            default_chunk_class = AIMessageChunk
            
            # Process each line in the response
            for line in response.iter_lines():
                if not line:
                    continue
                
                line_str = line.decode("utf-8")
                
                # Handle SSE format
                if line_str.startswith("data: "):
                    line_str = line_str[6:]
                    
                if line_str.strip() == "[DONE]":
                    break
                    
                try:
                    chunk_data = json.loads(line_str)
                    
                    # Process each choice in the chunk
                    if "choices" in chunk_data:
                        for choice in chunk_data["choices"]:
                            delta = choice.get("delta", {})
                            
                            # Get content from delta
                            content = delta.get("content", "")
                            
                            # Get role from delta if present
                            role = delta.get("role")
                            
                            # Handle tool calls if present
                            tool_calls = delta.get("tool_calls", [])
                            
                            # Create generation info if finish reason is present
                            generation_info = None
                            if finish_reason := choice.get("finish_reason"):
                                generation_info = {"finish_reason": finish_reason}
                            
                            # Create and yield the chunk
                            message_chunk = AIMessageChunk(
                                content=content,
                                role=role,
                                tool_calls=tool_calls if tool_calls else None,
                            )
                            
                            # Update default chunk class for future chunks
                            default_chunk_class = message_chunk.__class__
                            
                            yield message_chunk
                except json.JSONDecodeError as e:
                    if self.verbose:
                        print(f"Error decoding JSON: {e}")
                        print(f"Line: {line_str}")
                    continue
