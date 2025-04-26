"""ASI Chat wrapper."""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from typing import (
    Any, Dict, Iterator, List, Literal, Mapping, Optional, Sequence,
    Type, TypeVar, Union, cast
)

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.language_models.chat_models import agenerate_from_stream, generate_from_stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils import get_from_env, get_from_dict_or_env

logger = logging.getLogger(__name__)


class ChatASI(BaseChatModel):
    """ASI Chat large language models API.

    To use, you should have the environment variable ``ASI_API_KEY`` set,
    or pass it as a constructor parameter.

    Example:
        .. code-block:: python

            from langchain_asi import ChatASI
            chat = ChatASI(model_name="asi1-mini")
    
    The API base URL is https://api.asi1.ai/v1 for ASI1 models.
    """

    client: httpx.Client = Field(default=None, exclude=True)  #: :meta private:
    async_client: httpx.AsyncClient = Field(default=None, exclude=True)  #: :meta private:
    model_name: str
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    asi_api_key: Optional[str] = None
    """ASI API key."""
    asi_api_base: Optional[str] = None
    """Base URL path for API requests."""
    request_timeout: Union[float, tuple[float, float], Any, None] = None
    """Timeout for requests to ASI API."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    verbose: bool = False
    """Whether to print verbose output."""
    http_client: Optional[httpx.Client] = None
    """Optional httpx.Client for making requests."""
    http_async_client: Optional[httpx.AsyncClient] = None
    """Optional httpx.AsyncClient for making async requests."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["asi_api_key"] = get_from_dict_or_env(
            values, "asi_api_key", "ASI_API_KEY"
        )
        
        # Set the API base URL
        values["asi_api_base"] = get_from_dict_or_env(
            values, "asi_api_base", "ASI_API_BASE", "https://api.asi1.ai/v1"
        )

        # Initialize the httpx client if not provided
        if values.get("http_client") is None:
            values["http_client"] = httpx.Client()
        if values.get("async_client") is None:
            values["async_client"] = httpx.AsyncClient()

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "asi-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            **self.model_kwargs,
        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling ASI API."""
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        return params

    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for the API request."""
        headers = {
            "Authorization": f"Bearer {self.asi_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        return headers

    def _create_message_dicts(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Create message dicts for the API request."""
        message_dicts = []
        for message in messages:
            if isinstance(message, ChatMessage):
                message_dicts.append({
                    "role": message.role,
                    "content": message.content,
                })
            elif isinstance(message, HumanMessage):
                message_dicts.append({
                    "role": "user",
                    "content": message.content,
                })
            elif isinstance(message, AIMessage):
                message_dicts.append({
                    "role": "assistant",
                    "content": message.content,
                    **({
                        "tool_calls": message.additional_kwargs.get("tool_calls", [])
                    } if message.additional_kwargs.get("tool_calls") else {}),
                })
            elif isinstance(message, SystemMessage):
                message_dicts.append({
                    "role": "system",
                    "content": message.content,
                })
            elif isinstance(message, FunctionMessage):
                message_dicts.append({
                    "role": "function",
                    "content": message.content,
                    "name": message.name,
                })
            elif isinstance(message, ToolMessage):
                message_dicts.append({
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                })
            else:
                raise ValueError(f"Got unknown message type: {message}")

        return message_dicts

    def _create_chat_result(self, response: Dict[str, Any]) -> ChatResult:
        """Create a ChatResult from the response."""
        generations = []
        for choice in response.get("choices", []):
            message = choice.get("message", {})
            role = message.get("role")
            content = message.get("content")
            tool_calls = message.get("tool_calls", [])
            finish_reason = choice.get("finish_reason")

            # Create the appropriate message type
            if role == "assistant":
                additional_kwargs = {}
                
                # Add tool calls if present
                if tool_calls:
                    additional_kwargs["tool_calls"] = tool_calls
                
                # Add thought if present in response
                if "thought" in response:
                    additional_kwargs["thought"] = response.get("thought", [])
                
                ai_message = AIMessage(
                    content=content or "",
                    additional_kwargs=additional_kwargs if additional_kwargs else {},
                )

                # Add usage metadata if available
                if "usage" in response:
                    ai_message.usage_metadata = {
                        "input_tokens": response["usage"].get("prompt_tokens", 0),
                        "output_tokens": response["usage"].get("completion_tokens", 0),
                        "total_tokens": response["usage"].get("total_tokens", 0),
                    }

                # Create the generation
                gen = ChatGeneration(
                    message=ai_message,
                    generation_info={
                        "finish_reason": finish_reason
                    } if finish_reason else None,
                )
                generations.append(gen)

        # Return the chat result
        return ChatResult(
            generations=generations,
            llm_output={
                "token_usage": response.get("usage", {}),
                "model_name": response.get("model"),
                "thought": response.get("thought", []),
            }
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response.

        Args:
            messages: The messages to send to the chat model.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for tracking the run.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A ChatResult containing the generated responses.
        """
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        # Create message dicts and params for the API call
        message_dicts = self._create_message_dicts(messages)
        params = self._default_params.copy()
        params.update(kwargs)
        if stop:
            params["stop"] = stop

        # Set up the API request
        url = f"{self.asi_api_base}/chat/completions"
        headers = self._get_headers()

        # Prepare the request data
        data = {
            "model": self.model_name,
            "messages": message_dicts,
            **params,
        }

        # Log the request if verbose
        if self.verbose:
            print(f"Request URL: {url}")
            print(f"Request headers: {headers}")
            print(f"Request data: {data}")

        # Send the request
        response = self.client.post(
            url,
            headers=headers,
            json=data,
            timeout=self.request_timeout,
        )

        # Handle the response
        if response.status_code != 200:
            raise ValueError(
                f"Error from ASI API: {response.status_code} {response.text}"
            )

        # Parse the response
        response_data = response.json()
        return self._create_chat_result(response_data)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate a chat response.

        Args:
            messages: The messages to send to the chat model.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for tracking the run.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A ChatResult containing the generated responses.
        """
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        # Create message dicts and params for the API call
        message_dicts = self._create_message_dicts(messages)
        params = self._default_params.copy()
        params.update(kwargs)
        if stop:
            params["stop"] = stop

        # Set up the API request
        url = f"{self.asi_api_base}/chat/completions"
        headers = self._get_headers()

        # Prepare the request data
        data = {
            "model": self.model_name,
            "messages": message_dicts,
            **params,
        }

        # Log the request if verbose
        if self.verbose:
            print(f"Request URL: {url}")
            print(f"Request headers: {headers}")
            print(f"Request data: {data}")

        # Send the request
        response = await self.async_client.post(
            url,
            headers=headers,
            json=data,
            timeout=self.request_timeout,
        )

        # Handle the response
        if response.status_code != 200:
            raise ValueError(
                f"Error from ASI API: {response.status_code} {response.text}"
            )

        # Parse the response
        response_data = response.json()
        return self._create_chat_result(response_data)

    def _process_chunk(self, chunk_data: Dict[str, Any]) -> Optional[ChatGenerationChunk]:
        """Process a chunk from the streaming response.

        Args:
            chunk_data: The chunk data to process.

        Returns:
            A ChatGenerationChunk or None if the chunk should be skipped.
        """
        if "choices" not in chunk_data or not chunk_data["choices"]:
            return None

        choice = chunk_data["choices"][0]
        delta = choice.get("delta", {})
        role = delta.get("role")
        content = delta.get("content", "")
        tool_calls = delta.get("tool_calls", [])
        finish_reason = choice.get("finish_reason")

        # Skip role-only deltas (they don't contain content)
        if role and not content and not tool_calls and not finish_reason:
            return None

        # Create the appropriate message chunk
        message_kwargs = {}
        if tool_calls:
            message_kwargs["tool_calls"] = tool_calls
            
        # Add thought if present in chunk
        if "thought" in chunk_data:
            message_kwargs["thought"] = chunk_data.get("thought", [])

        message = AIMessageChunk(
            content=content or "",
            additional_kwargs=message_kwargs if message_kwargs else None
        )

        # Create the generation chunk
        return ChatGenerationChunk(
            message=message,
            generation_info={
                "finish_reason": finish_reason
            } if finish_reason else None,
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the chat response.

        Args:
            messages: The messages to send to the chat model.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for tracking the run.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            An iterator of ChatGenerationChunk objects.
        """
        # Create message dicts and params for the API call
        message_dicts = self._create_message_dicts(messages)
        params = self._default_params.copy()
        params.update(kwargs)
        params["stream"] = True
        if stop:
            params["stop"] = stop

        # Set up the API request
        url = f"{self.asi_api_base}/chat/completions"
        headers = self._get_headers()

        # Prepare the request data
        data = {
            "model": self.model_name,
            "messages": message_dicts,
            **params,
        }

        # Log the request if verbose
        if self.verbose:
            print(f"Request URL: {url}")
            print(f"Request headers: {headers}")
            print(f"Request data: {data}")

        # Send the request
        with self.client.stream(
            "POST",
            url,
            headers=headers,
            json=data,
            timeout=self.request_timeout,
        ) as response:
            if response.status_code != 200:
                raise ValueError(
                    f"Error from ASI API: {response.status_code} {response.text}"
                )

            # Process the streaming response
            for line in response.iter_lines():
                if not line:
                    continue

                # Decode the line if it's bytes
                if isinstance(line, bytes):
                    line_str = line.decode("utf-8")
                else:
                    line_str = line

                # Skip the "data: " prefix if present
                if line_str.startswith("data: "):
                    line_str = line_str[6:]

                # Skip empty lines and [DONE] marker
                if not line_str or line_str == "[DONE]":
                    continue

                # Parse the JSON response
                try:
                    chunk_data = json.loads(line_str)
                    chunk = self._process_chunk(chunk_data)
                    if chunk is not None:
                        if run_manager:
                            run_manager.on_llm_new_token(
                                chunk.message.content or "",
                                verbose=self.verbose,
                            )
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
        """Asynchronously stream the chat response.

        Args:
            messages: The messages to send to the chat model.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for tracking the run.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            An async iterator of ChatGenerationChunk objects.
        """
        # Create message dicts and params for the API call
        message_dicts = self._create_message_dicts(messages)
        params = self._default_params.copy()
        params.update(kwargs)
        params["stream"] = True
        if stop:
            params["stop"] = stop

        # Set up the API request
        url = f"{self.asi_api_base}/chat/completions"
        headers = self._get_headers()

        # Prepare the request data
        data = {
            "model": self.model_name,
            "messages": message_dicts,
            **params,
        }

        # Log the request if verbose
        if self.verbose:
            print(f"Request URL: {url}")
            print(f"Request headers: {headers}")
            print(f"Request data: {data}")

        # Send the request
        async with self.async_client.stream(
            "POST",
            url,
            headers=headers,
            json=data,
            timeout=self.request_timeout,
        ) as response:
            if response.status_code != 200:
                raise ValueError(
                    f"Error from ASI API: {response.status_code} {response.text}"
                )

            # Process the streaming response
            async for line in response.aiter_lines():
                if not line:
                    continue

                # Skip the "data: " prefix if present
                if line.startswith("data: "):
                    line = line[6:]

                # Skip empty lines and [DONE] marker
                if not line or line == "[DONE]":
                    continue

                # Parse the JSON response
                try:
                    chunk_data = json.loads(line)
                    chunk = self._process_chunk(chunk_data)
                    if chunk is not None:
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                chunk.message.content or "",
                                verbose=self.verbose,
                            )
                        yield chunk
                except json.JSONDecodeError as e:
                    if self.verbose:
                        print(f"Error decoding JSON: {e}")
                        print(f"Line: {line}")
                    continue

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

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        This is an approximation since we don't have access to ASI's tokenizer.
        We use a simple heuristic based on whitespace tokenization.

        Args:
            text: The text to count tokens for.

        Returns:
            The approximate number of tokens in the text.
        """
        # Simple approximation: count words and multiply by 1.3 (average tokens per word)
        return int(len(text.split()) * 1.3)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            The number of tokens in the text.
        """
        return self._count_tokens(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in a list of messages.

        Args:
            messages: The messages to count tokens for.

        Returns:
            The number of tokens in the messages.
        """
        message_dicts = self._create_message_dicts(messages)
        text = json.dumps(message_dicts)
        return self._count_tokens(text)
