from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.document import Document
from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGenerationChunk, GenerationChunk, LLMResult
from tenacity import RetryCallState
from opentelemetry import trace

from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID


class OpentelemetryCallback(AsyncCallbackHandler):
    """Callback handler to submit events to opentelemetry"""

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_llm_start",
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_chat_model_start",
            attributes={
                "chatbot.llms.id": "_".join(serialized.get("id")),
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_llm_new_token",
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_llm_end",
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        current_span = trace.get_current_span()
        current_span.record_exception(
            error,
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_chain_start",
            attributes={
                "chatbot.llms.id": "_".join(serialized.get("id")),
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.question": inputs.get("question", "no question found"),
                "chatbot.llms.documents": self.get_documents_metadata_from_input(
                    inputs.get("input_documents", [])
                ),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    def get_documents_metadata_from_input(self, documents: List[Document]):
        return ",".join([document.metadata.get("name") for document in documents])

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_chain_end",
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        current_span = trace.get_current_span()
        current_span.record_exception(
            error,
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_tool_start",
            attributes={
                "chatbot.llms.id": "_".join(serialized.get("id")),
                "chatbot.llms.input_string": input_str,
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_tool_end",
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        current_span = trace.get_current_span()
        current_span.record_exception(
            error,
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on arbitrary text."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_text",
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on a retry event."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_retry",
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_agent_action",
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent end."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_agent_finish",
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on retriever start."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_retriever_start",
            attributes={
                "chatbot.llms.id": "_".join(serialized.get("id")),
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on retriever end."""
        current_span = trace.get_current_span()
        current_span.add_event(
            "on_retriever_end",
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )

    async def on_retriever_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on retriever error."""
        current_span = trace.get_current_span()
        current_span.record_exception(
            error,
            attributes={
                "chatbot.llms.run_id": str(run_id),
                "chatbot.llms.parent_run_id": str(parent_run_id),
            },
        )
