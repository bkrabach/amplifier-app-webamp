"""
Amplifier Browser Adapter - Proof of Concept

This module demonstrates how Amplifier's core patterns could run in the browser
via Pyodide, with WebGPU providing local LLM inference.

Key concepts demonstrated:
- Session management (history, context)
- Provider abstraction (WebGPU provider via JS bridge)
- Hook-style events (simplified)
- Message models (Amplifier-compatible structure)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TYPE_CHECKING
from enum import Enum
import inspect
import json

# Type aliases for cleaner annotations
HookCallback = Callable[[Any], None]
ChunkCallback = Callable[[str], None]
MessageDict = dict[str, Any]

# These functions are injected by JavaScript at runtime via Pyodide globals
# We declare them here for type checking purposes
if TYPE_CHECKING:

    async def js_llm_complete(messages_json: str) -> str: ...
    async def js_llm_stream(messages_json: str, on_chunk: ChunkCallback) -> str: ...


# =============================================================================
# Message Models (simplified from amplifier-core)
# =============================================================================


class Role(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A chat message."""

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> MessageDict:
        d: MessageDict = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class Usage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CompletionResult:
    """Result from a completion request."""

    content: str
    usage: Usage
    model: str = ""
    finish_reason: str = "stop"


# =============================================================================
# Provider Protocol (from amplifier-core interfaces)
# =============================================================================


class Provider(Protocol):
    """Protocol for LLM providers."""

    @property
    def name(self) -> str:
        """Provider identifier."""
        ...

    async def complete(
        self, messages: list[Message], **kwargs: Any
    ) -> CompletionResult:
        """Generate a completion."""
        ...

    async def stream(
        self, messages: list[Message], on_chunk: ChunkCallback, **kwargs: Any
    ) -> CompletionResult:
        """Generate a streaming completion."""
        ...


# =============================================================================
# WebGPU Provider (bridges to JavaScript WebLLM)
# =============================================================================


class WebGPUProvider:
    """
    Provider that uses WebGPU via JavaScript bridge.

    This provider calls JavaScript functions exposed by Pyodide
    to perform inference using WebLLM.
    """

    def __init__(self, model_id: str = "Phi-3.5-mini-instruct"):
        self.model_id = model_id
        self._name = "webgpu"

    @property
    def name(self) -> str:
        return self._name

    async def complete(
        self, messages: list[Message], **kwargs: Any
    ) -> CompletionResult:
        """Generate a completion using WebGPU."""
        # Convert messages to JSON for JS bridge
        messages_json = json.dumps([m.to_dict() for m in messages])

        # Call JavaScript LLM function
        # js_llm_complete is injected by main.js
        result_json = await js_llm_complete(messages_json)
        result = json.loads(result_json)

        return CompletionResult(
            content=result["content"],
            usage=Usage(
                prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                total_tokens=result.get("usage", {}).get("total_tokens", 0),
            ),
            model=self.model_id,
        )

    async def stream(
        self,
        messages: list[Message],
        on_chunk: ChunkCallback,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a streaming completion using WebGPU."""
        # pyodide.ffi is only available in browser runtime
        # Import dynamically to allow type checking outside browser
        try:
            from pyodide.ffi import create_proxy  # type: ignore[import-not-found]
        except ImportError:
            # Fallback for non-browser environments (testing)
            def create_proxy(fn: ChunkCallback) -> ChunkCallback:
                return fn

        messages_json = json.dumps([m.to_dict() for m in messages])

        # Create a proxy for the callback (pyodide returns Any, so cast it)
        chunk_callback: ChunkCallback = create_proxy(on_chunk)  # pyright: ignore[reportUnknownVariableType]

        # Call JavaScript streaming LLM function
        result_json = await js_llm_stream(messages_json, chunk_callback)  # pyright: ignore[reportUnknownArgumentType]
        result = json.loads(result_json)

        return CompletionResult(
            content=result["content"],
            usage=Usage(
                prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                total_tokens=result.get("usage", {}).get("total_tokens", 0),
            ),
            model=self.model_id,
        )


# =============================================================================
# Hook System (simplified from amplifier-core)
# =============================================================================


class HookType(str, Enum):
    """Types of hooks that can be registered."""

    BEFORE_COMPLETION = "before_completion"
    AFTER_COMPLETION = "after_completion"
    ON_ERROR = "on_error"
    ON_STREAM_CHUNK = "on_stream_chunk"


@dataclass
class HookRegistry:
    """Registry for session hooks."""

    _hooks: dict[HookType, list[HookCallback]] = field(
        default_factory=lambda: {}  # pyright: ignore[reportUnknownLambdaType]
    )

    def register(self, hook_type: HookType, callback: HookCallback) -> None:
        """Register a hook callback."""
        if hook_type not in self._hooks:
            self._hooks[hook_type] = []
        self._hooks[hook_type].append(callback)

    async def emit(self, hook_type: HookType, data: Any = None):
        """Emit an event to all registered hooks."""
        if hook_type in self._hooks:
            for callback in self._hooks[hook_type]:
                if inspect.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)


# =============================================================================
# Context Manager (simplified)
# =============================================================================


@dataclass
class ContextManager:
    """
    Manages conversation context and history.

    In a full implementation, this would handle:
    - Token budget management
    - Context window truncation
    - System prompt injection
    - RAG context insertion
    """

    max_history: int = 50
    _history: list[Message] = field(
        default_factory=lambda: []  # pyright: ignore[reportUnknownLambdaType]
    )

    def add_message(self, message: Message):
        """Add a message to history."""
        self._history.append(message)

        # Trim if exceeds max
        if len(self._history) > self.max_history:
            # Keep system messages and trim oldest user/assistant
            self._history = self._history[-self.max_history :]

    def get_messages(self, system_prompt: str | None = None) -> list[Message]:
        """Get all messages including optional system prompt."""
        messages: list[Message] = []

        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))

        messages.extend(self._history)
        return messages

    def clear(self):
        """Clear conversation history."""
        self._history = []


# =============================================================================
# Amplifier Browser Session
# =============================================================================


class AmplifierBrowserSession:
    """
    Browser-compatible Amplifier session.

    This is a simplified version of AmplifierSession that demonstrates
    the core patterns while running entirely in the browser.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant running entirely in the user's browser.

You are powered by a local language model via WebGPU - no data is sent to external servers.
Be concise, helpful, and friendly. If you don't know something, say so honestly."""

    def __init__(
        self,
        provider: Provider | None = None,
        system_prompt: str | None = None,
    ):
        self.provider = provider or WebGPUProvider()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.context = ContextManager()
        self.hooks = HookRegistry()

        # Simple history for JS interop
        self.history: list[MessageDict] = []

        print(
            f"AmplifierBrowserSession initialized with provider: {self.provider.name}"
        )

    async def execute(
        self,
        prompt: str,
        stream: bool = True,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """
        Execute a prompt and return the response.

        Args:
            prompt: User's input message
            stream: Whether to stream the response
            on_chunk: Callback for each streamed chunk

        Returns:
            The assistant's response
        """
        # Create user message
        user_message = Message(role=Role.USER, content=prompt)
        self.context.add_message(user_message)
        self.history.append({"role": "user", "content": prompt})

        # Emit before hook
        await self.hooks.emit(
            HookType.BEFORE_COMPLETION,
            {
                "prompt": prompt,
                "history_length": len(self.history),
            },
        )

        try:
            # Get all messages including system prompt
            messages = self.context.get_messages(self.system_prompt)

            # Generate completion
            if stream and on_chunk:
                result = await self.provider.stream(messages, on_chunk)
            else:
                result = await self.provider.complete(messages)

            # Add assistant response to context
            assistant_message = Message(role=Role.ASSISTANT, content=result.content)
            self.context.add_message(assistant_message)
            self.history.append({"role": "assistant", "content": result.content})

            # Emit after hook
            await self.hooks.emit(
                HookType.AFTER_COMPLETION,
                {
                    "response": result.content,
                    "usage": result.usage,
                },
            )

            return result.content

        except Exception as e:
            await self.hooks.emit(HookType.ON_ERROR, {"error": str(e)})
            raise

    def clear_history(self):
        """Clear conversation history."""
        self.context.clear()
        self.history = []
        print("Conversation history cleared")

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt
        print(f"System prompt updated ({len(prompt)} chars)")


# =============================================================================
# RAG Support (placeholder for future)
# =============================================================================


@dataclass
class Document:
    """A document chunk for RAG."""

    content: str
    metadata: dict[str, Any] = field(
        default_factory=lambda: {}  # pyright: ignore[reportUnknownLambdaType]
    )
    embedding: list[float] | None = None


class VectorStore(Protocol):
    """Protocol for vector stores."""

    async def add(self, documents: list[Document]) -> None:
        """Add documents to the store."""
        ...

    async def search(self, query: str, k: int = 3) -> list[Document]:
        """Search for similar documents."""
        ...


class InMemoryVectorStore:
    """
    Simple in-memory vector store.

    In a full implementation, this would:
    - Use IndexedDB for persistence
    - Use Transformers.js for embeddings
    - Support efficient similarity search
    """

    def __init__(self):
        self.documents: list[Document] = []

    async def add(self, documents: list[Document]) -> None:
        """Add documents (embedding would happen here)."""
        self.documents.extend(documents)
        print(f"Added {len(documents)} documents to vector store")

    async def search(self, query: str, k: int = 3) -> list[Document]:
        """
        Search for similar documents.

        TODO: Implement actual similarity search with embeddings.
        For now, returns most recent documents.
        """
        return self.documents[-k:] if self.documents else []


# =============================================================================
# Utility Functions
# =============================================================================


def format_rag_context(documents: list[Document]) -> str:
    """Format retrieved documents as context for the LLM."""
    if not documents:
        return ""

    context_parts = ["Here is relevant context:\n"]
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"[{i}] {doc.content}\n")

    return "\n".join(context_parts)


# =============================================================================
# Demo / Test
# =============================================================================


async def demo():
    """Quick demo of the session."""
    session = AmplifierBrowserSession()

    # Register a simple hook
    def on_completion(data: Any) -> None:
        print(f"Completion finished. Usage: {data.get('usage')}")

    session.hooks.register(HookType.AFTER_COMPLETION, on_completion)

    # Execute a prompt
    response = await session.execute("Hello! What can you help me with?")
    print(f"Response: {response[:100]}...")

    return session


# Export for use
__all__ = [
    "AmplifierBrowserSession",
    "WebGPUProvider",
    "Message",
    "Role",
    "HookType",
    "Document",
    "InMemoryVectorStore",
]

print("Amplifier Browser Adapter loaded successfully!")
