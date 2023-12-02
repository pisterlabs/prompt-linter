from pathlib import Path
from typing import Dict, Iterator, List, cast

from llama_index import ServiceContext, StorageContext
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.indices.loading import load_index_from_storage
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.llms import ChatMessage, MessageRole, OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.schema import NodeWithScore
from pydantic import BaseModel, Field


class ParentalLeaveStatuteQuery(BaseModel):
    """Represents a query for the parental leave statute."""

    question: str = Field(description="The question to ask the parental leave statute.")
    """The question to ask the parental leave statute."""

    situational_context: Dict[str, str] = Field(default_factory=dict)
    """User's situational context as key-value pairs.

    The keys are the names of the situational context variables and the values are the
    values of the situational context variables. The names are descriptions like
    "arbejdsforhold" and "arbejdstimer" and the values are the actual values like
    "lønmodtager" and "37 timer om ugen".
    """


class ParentalLeaveStatuteQAEngine:
    """Represents a question-answering engine for the parental leave statute."""

    def __init__(
        self,
        index_dir: Path,
        cohere_api_key: str,
        openai_api_key: str,
        llm_model_name: str = "gpt-4",
    ) -> None:
        # TODO: Refactor this.
        self._llm = OpenAI(
            api_key=openai_api_key,
            model=llm_model_name,
            temperature=0.0,
        )

        self._messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "Dit navn er Lærbar. Du er jura-professor, "
                    "der er ekspert i barselsloven. Du hjælper folk med "
                    "at forstå barselsloven og besvare spørgsmål om barselsloven. "
                    "Dine svar er baseret på tekst-fraser citeret direkte "
                    "fra barselsloven."
                ),
            )
        ]

        embed_model = CohereEmbedding(
            cohere_api_key=cohere_api_key,
            model_name="embed-multilingual-v3.0",
            input_type="search_query",
        )

        node_parser: SimpleNodeParser = SimpleNodeParser.from_defaults(
            chunk_size=512,
            chunk_overlap=10,
        )

        self._service_context: ServiceContext = ServiceContext.from_defaults(
            llm=None,
            embed_model=embed_model,
            node_parser=node_parser,
        )

        base_index = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=str(index_dir)),
            service_context=self._service_context,
        )

        self._vector_index: VectorStoreIndex = cast(VectorStoreIndex, base_index)

        # Configure the response mode so the retriever only returns the nodes
        # without sending the retreived nodes to an LLM.
        # https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#configuring-the-response-mode
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.NO_TEXT,
            service_context=self._service_context,
        )

        base_retriever = self._vector_index.as_retriever(
            service_context=self._service_context,
            response_synthesizer=response_synthesizer,
        )

        self._retriever: VectorIndexRetriever = cast(
            VectorIndexRetriever, base_retriever
        )

    def run(self, query: ParentalLeaveStatuteQuery) -> Iterator[str]:
        query_for_retriever = self._build_query_for_retriever(query=query)

        retrieved_nodes = self._retriever.retrieve(
            str_or_query_bundle=query_for_retriever,
        )

        llm_prompt = self._build_llm_prompt(
            query=query, retrieved_nodes=retrieved_nodes
        )
        print(llm_prompt)

        for item in self._stream_llm_response(llm_prompt=llm_prompt):
            yield item

        yield "\n\n### Kilder\n\n"
        for item in self._stream_retreived_nodes(retrieved_nodes=retrieved_nodes):
            yield item

    def _stream_retreived_nodes(
        self, retrieved_nodes: List[NodeWithScore]
    ) -> Iterator[str]:
        for source_node in retrieved_nodes:
            # source_text_fmt = source_node.node.get_content(metadata_mode=MetadataMode.ALL).strip()

            reference = source_node.node.metadata["Reference"]
            chapter_no = source_node.node.metadata["Kapitel nummer"]
            chapter_title = source_node.node.metadata["Kapitel overskrift"]
            is_paragraph = source_node.node.metadata.get("Type", "") == "Paragraf"
            short_guid = source_node.node_id.split("-")[0]

            yield f"**Kapitel {chapter_no}: {chapter_title}."

            if is_paragraph:
                yield f" Paragraf: {reference}"
            else:
                yield f" {reference}"
            yield f"** [{short_guid}]\n\n"

            yield f"{source_node.node.get_content().strip()}\n\n"

    def _stream_llm_response(self, llm_prompt: str) -> Iterator[str]:
        """Query the LLM and stream the response.

        Args:
            llm_prompt (str): The prompt for the LLM.

        Yields:
            Iterator[str]: The response from the LLM.
        """
        self._messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=llm_prompt,
            )
        )

        llm_completion_resp = self._llm.stream_chat(
            messages=self._messages,
        )

        full_response = ""
        for chunk in llm_completion_resp:
            chunk_text = chunk.delta
            full_response += chunk_text
            yield chunk_text

        self._messages.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=full_response,
            )
        )

        print(f"Full response:\n\n{full_response}")

    def _build_llm_prompt(
        self, query: ParentalLeaveStatuteQuery, retrieved_nodes: List[NodeWithScore]
    ) -> str:
        """Build the prompt for the query."""
        prompt = ""

        prompt += "Du får et spørgsmål fra en person, hvis situation ser sådan ud:\n\n"
        for key, value in query.situational_context.items():
            prompt += f" - {key}: {value}\n"

        prompt += "\n"
        prompt += "Personen stiller flg. spørgsmål:\n\n"
        prompt += f"{query.question}\n\n"

        if len(retrieved_nodes) > 0:
            prompt += "## Kilder\n\n"
            prompt += "Et opslag i barselsloven giver flg. tekster.\n\n"
            for source_node in retrieved_nodes:
                reference = source_node.node.metadata["Reference"]
                chapter_no = source_node.node.metadata["Kapitel nummer"]
                chapter_title = source_node.node.metadata["Kapitel overskrift"]
                is_paragraph = source_node.node.metadata.get("Type", "") == "Paragraf"
                short_guid = source_node.node_id.split("-")[0]

                source_text = (
                    f"### [{short_guid}] Kapitel {chapter_no}: {chapter_title}."
                )
                if is_paragraph:
                    source_text += f" Paragraf: {reference}"
                else:
                    source_text += f" {reference}"

                prompt += f"{source_text}\n\n"
                prompt += f"{source_node.node.get_content().strip()}\n\n"

        prompt += "Din opgave er bevare konteksten fra spørgsmålet og svare på spørgsmålet med en kort tekst. "
        prompt += "Dit svar skal altid inkludere en eller flere referencer fra Kilder-sektionen.\n"

        return prompt

    def _build_query_for_retriever(self, query: ParentalLeaveStatuteQuery) -> str:
        """Build the query for the retriever.

        The query is the question with the situational context as a prefix.

        Args:
            query (ParentalLeaveStatuteQuery): The query.

        Returns:
            str: The query for the retriever.
        """

        question_with_context = ""

        if len(query.situational_context) > 0:
            question_with_context += "Min nuværende situtation er:\n"

            for key, value in query.situational_context.items():
                question_with_context += f" - {key}: {value}\n"

            question_with_context += "\n"

        question_with_context += "Mit spørgsmål er:\n"
        question_with_context += query.question

        return question_with_context
