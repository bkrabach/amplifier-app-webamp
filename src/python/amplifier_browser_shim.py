"""
Amplifier Browser Shim - Enables amplifier-core to run in Pyodide.

This module provides browser-compatible implementations that bypass
filesystem-dependent features while preserving the full power of
amplifier-core's type system, hooks, and coordinator infrastructure.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, AsyncIterator, TYPE_CHECKING

# amplifier-core imports - these resolve at runtime in Pyodide after micropip install
# pyright: reportMissingImports=false
from amplifier_core.interfaces import Provider, ContextManager  # type: ignore[import-not-found]
from amplifier_core.message_models import (  # type: ignore[import-not-found]
    ChatRequest,
    ChatResponse,
    Message,
    Usage,
    TextBlock,
)
from amplifier_core.models import ProviderInfo, ModelInfo  # type: ignore[import-not-found]

# Note: We don't use ModuleCoordinator in browser - it requires filesystem access
from amplifier_core.hooks import HookRegistry  # type: ignore[import-not-found]
from amplifier_core import events  # type: ignore[import-not-found]

# Role constants (amplifier-core uses Literal strings, not enums)
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# JS bridge functions injected by main.js
if TYPE_CHECKING:

    async def js_llm_complete(messages_json: str) -> str: ...
    async def js_llm_stream(
        messages_json: str, on_chunk: Callable[[str], None]
    ) -> str: ...


logger = logging.getLogger(__name__)


# =============================================================================
# WebGPU Provider - Bridges to JavaScript WebLLM
# =============================================================================


class WebGPUProvider(Provider):
    """
    Provider that uses WebGPU via JavaScript bridge.

    Implements the full amplifier-core Provider protocol while
    delegating actual inference to WebLLM running in JavaScript.
    """

    def __init__(self, model_id: str = "Phi-3.5-mini-instruct"):
        self.model_id = model_id
        self._name = "webgpu"

    @property
    def name(self) -> str:
        return self._name

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="webgpu",
            display_name="WebGPU Local LLM",
            description="Local LLM inference via WebGPU - runs entirely in browser",
            supports_streaming=True,
            supports_tools=False,  # TODO: Add tool support later
            supports_vision=False,
        )

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=self.model_id,
                display_name="Phi-3.5 Mini Instruct",
                context_window=128000,
                supports_tools=False,
                supports_vision=False,
            )
        ]

    async def complete(
        self,
        request: ChatRequest,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a completion using WebGPU."""
        # Convert messages to format expected by JS bridge
        messages_data = []
        for msg in request.messages:
            msg_dict = {"role": msg.role.value, "content": msg.content or ""}
            if msg.name:
                msg_dict["name"] = msg.name
            messages_data.append(msg_dict)

        messages_json = json.dumps(messages_data)

        # Call JavaScript LLM function
        result_json = await js_llm_complete(messages_json)
        result = json.loads(result_json)

        # ChatResponse.content is a list of ContentBlocks
        content_text = result.get("content", "")
        return ChatResponse(
            content=[TextBlock(text=content_text)],
            usage=Usage(
                prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                total_tokens=result.get("usage", {}).get("total_tokens", 0),
            ),
            finish_reason="stop",
        )

    async def stream(
        self,
        request: ChatRequest,
        **kwargs: Any,
    ) -> AsyncIterator[ChatResponse]:
        """Generate a streaming completion using WebGPU."""
        # For now, fall back to non-streaming
        # TODO: Implement proper streaming with JS bridge
        response = await self.complete(request, **kwargs)
        yield response


# =============================================================================
# Simple Context Manager - In-memory message storage
# =============================================================================


class SimpleContextManager(ContextManager):
    """
    Simple in-memory context manager for browser use.

    Stores messages in memory with basic truncation support.
    """

    def __init__(self, max_messages: int = 100):
        self._messages: list[Message] = []
        self._max_messages = max_messages
        self._system_prompt: str | None = None

    async def add_message(self, message: Message) -> None:
        """Add a message to context."""
        self._messages.append(message)

        # Trim if needed (keep system messages)
        if len(self._messages) > self._max_messages:
            system_msgs = [m for m in self._messages if m.role == ROLE_SYSTEM]
            other_msgs = [m for m in self._messages if m.role != ROLE_SYSTEM]
            keep_count = self._max_messages - len(system_msgs)
            self._messages = system_msgs + other_msgs[-keep_count:]

    async def get_messages(self) -> list[Message]:
        """Get all messages."""
        return list(self._messages)

    async def get_messages_for_request(
        self,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[Message]:
        """Get messages formatted for a chat request."""
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append(Message(role=ROLE_SYSTEM, content=system_prompt))
        elif self._system_prompt:
            messages.append(Message(role=ROLE_SYSTEM, content=self._system_prompt))

        # Add conversation history (excluding any existing system messages in history)
        for msg in self._messages:
            if msg.role != ROLE_SYSTEM:
                messages.append(msg)

        return messages

    async def set_messages(self, messages: list[Message]) -> None:
        """Replace all messages."""
        self._messages = list(messages)

    async def clear(self) -> None:
        """Clear all messages."""
        self._messages = []

    def set_system_prompt(self, prompt: str) -> None:
        """Set the default system prompt."""
        self._system_prompt = prompt


# Note: SimpleOrchestrator and BrowserModuleLoader removed - not needed
# The BrowserAmplifierSession handles orchestration directly


# =============================================================================
# Browser Session - Ties everything together
# =============================================================================


class BrowserAmplifierSession:
    """
    Browser-compatible Amplifier session.

    Simplified session that uses amplifier-core's type system but bypasses
    the kernel's ModuleCoordinator infrastructure which requires filesystem access.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant running entirely in the user's browser.

You are powered by a local language model via WebGPU - no data is sent to external servers.
Be concise, helpful, and friendly. If you don't know something, say so honestly."""

    def __init__(
        self,
        model_id: str = "Phi-3.5-mini-instruct",
        system_prompt: str | None = None,
    ):
        self.model_id = model_id
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Create browser-native modules (using amplifier-core interfaces)
        self.provider = WebGPUProvider(model_id=model_id)
        self.context = SimpleContextManager()
        self.hooks = HookRegistry()

        # Simple config dict (not using ModuleCoordinator)
        self.config = {
            "system_prompt": self.system_prompt,
            "model_id": model_id,
        }

        # History list for JS interop (simple list of dicts)
        self.history: list[dict[str, str]] = []

        self._initialized = False
        print(f"BrowserAmplifierSession created with model: {model_id}")

    async def initialize(self) -> None:
        """Initialize the session."""
        if self._initialized:
            return

        # Set system prompt on context
        self.context.set_system_prompt(self.system_prompt)

        self._initialized = True
        print("BrowserAmplifierSession initialized")

    async def execute(
        self,
        prompt: str,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """
        Execute a prompt and return the response.

        Uses amplifier-core types but simple direct orchestration.

        Args:
            prompt: User's input message
            stream: Whether to stream (not yet implemented)
            on_chunk: Callback for streaming chunks

        Returns:
            The assistant's response text
        """
        if not self._initialized:
            await self.initialize()

        # Add user message to context
        user_message = Message(role=ROLE_USER, content=prompt)
        await self.context.add_message(user_message)

        # Emit hook event
        await self.hooks.emit(events.PROVIDER_REQUEST, {"message": prompt})

        try:
            # Get messages for request
            messages = await self.context.get_messages_for_request(
                system_prompt=self.system_prompt
            )

            # Create chat request using amplifier-core types
            request = ChatRequest(messages=messages)

            # Get completion from provider
            response = await self.provider.complete(request)

            # Extract text from response blocks
            response_text = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        response_text += block.text
                    elif isinstance(block, str):
                        response_text += block

            # Add assistant response to context
            assistant_message = Message(role=ROLE_ASSISTANT, content=response_text)
            await self.context.add_message(assistant_message)

            # Emit hook event
            await self.hooks.emit(
                events.PROVIDER_RESPONSE,
                {"response": response_text, "usage": response.usage},
            )

            return response_text

        except Exception as e:
            await self.hooks.emit(events.PROVIDER_ERROR, {"error": str(e)})
            raise

    async def get_history(self) -> list[dict[str, Any]]:
        """Get conversation history as dicts for JS interop."""
        messages = await self.context.get_messages()
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    async def clear_history(self) -> None:
        """Clear conversation history."""
        self.context._messages = []
        print("Conversation history cleared")

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt."""
        self.system_prompt = prompt
        self.context.set_system_prompt(prompt)
        self.config["system_prompt"] = prompt
        print(f"System prompt updated ({len(prompt)} chars)")


# =============================================================================
# Factory function for JavaScript
# =============================================================================


def create_session(
    model_id: str = "Phi-3.5-mini-instruct",
    system_prompt: str | None = None,
) -> BrowserAmplifierSession:
    """
    Create a new browser Amplifier session.

    This is the main entry point called from JavaScript.
    """
    return BrowserAmplifierSession(
        model_id=model_id,
        system_prompt=system_prompt,
    )


# Export for use
__all__ = [
    "BrowserAmplifierSession",
    "WebGPUProvider",
    "SimpleContextManager",
    "create_session",
]

print("Amplifier Browser Shim loaded!")
