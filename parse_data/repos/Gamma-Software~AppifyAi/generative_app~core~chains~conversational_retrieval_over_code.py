"""Chain for chatting with a vector database."""
from __future__ import annotations

import inspect
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseRetriever, Document

from langchain.chains.conversational_retrieval.base import (
    CHAT_TURN_TYPE,
    _get_chat_history,
)

from chains.prompt import (
    CONDENSE_QUESTION_CODE_PROMPT,
    PROMPT,
    prompt_missing_imports_check,
)
from utils.security import analyze_security
from chains.parser import parse_code


def remove_entrypoint(code):
    lines = code.split("\n")
    modified_lines = []
    entrypoint_found = False
    for line in lines:
        if (
            line.strip() == 'if __name__ == "__main__":'
            or line.strip() == "if __name__ == '__main__':"
        ):
            entrypoint_found = True
        elif entrypoint_found:
            modified_lines.append(line.lstrip())
        else:
            modified_lines.append(line)

    modified_code = "\n".join(modified_lines)
    modified_code = modified_code.rstrip()
    return modified_code


class BaseConversationalRetrievalCodeChain(Chain):
    """Chain for chatting with an index. Given the chat history,
    the current code and a question, return the answer."""

    combine_docs_chain: BaseCombineDocumentsChain
    question_generator: LLMChain
    missing_imports_chain: LLMChain
    output_key: List[str] = ["code", "explanation"]
    return_source_documents: bool = False
    return_generated_question: bool = False
    return_revision_request: bool = False
    get_chat_history: Optional[Callable[[CHAT_TURN_TYPE], str]] = None
    """Return the source documents."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = ["code", "explanation"]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        if self.return_generated_question:
            _output_keys = _output_keys + ["generated_question"]
        return _output_keys

    @abstractmethod
    def _get_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        request = inputs["question"]
        new_request = self.question_generator.run(
            question=request, callbacks=_run_manager.get_child()
        )
        new_request = None if "None" in new_request else new_request
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if new_request is not None:
            if accepts_run_manager:
                docs = self._get_docs(new_request, inputs, run_manager=_run_manager)
            else:
                docs = self._get_docs(new_request, inputs)  # type: ignore[call-arg]
        else:
            docs = []

        new_inputs = inputs.copy()
        # Remove any mentions of streamlit or python from the question
        if new_request is not None:
            new_request = new_request.replace("streamlit", "").replace("python", "")
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        new_inputs["chat_history"] = chat_history_str
        answer = self.combine_docs_chain.run(
            input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs
        )
        code, expl = parse_code(answer)

        is_code_not_safe = False
        if code is not None:
            # Run check code
            is_code_not_safe = analyze_security(code)
            if not is_code_not_safe:
                code = remove_entrypoint(code)
                # Check if imports are missing
                code_checked = self.missing_imports_chain.run(code=code)
                code_checked = None if code_checked == "None" else code_checked
                if code_checked is not None:
                    code = code_checked

        output: Dict[str, Any] = {self.output_key[0]: code, self.output_key[1]: expl}
        if self.return_source_documents:
            output["source_documents"] = docs
        if self.return_generated_question:
            output["generated_question"] = new_request
        if self.return_revision_request:
            output["revision_request"] = is_code_not_safe
        return output

    @abstractmethod
    async def _aget_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        request = inputs["question"]
        new_request = await self.question_generator.arun(
            question=request, callbacks=_run_manager.get_child()
        )
        new_request = None if "None" in new_request else new_request
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if new_request is not None:
            if accepts_run_manager:
                docs = await self._aget_docs(
                    new_request, inputs, run_manager=_run_manager
                )
            else:
                docs = await self._aget_docs(new_request, inputs)  # type: ignore[call-arg]
        else:
            docs = []

        new_inputs = inputs.copy()
        # Remove any mentions of streamlit or python from the question
        if new_request is not None:
            new_request = new_request.replace("streamlit", "").replace("python", "")
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        new_inputs["chat_history"] = chat_history_str
        answer = await self.combine_docs_chain.arun(
            input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs
        )
        code, expl = parse_code(answer)

        is_code_not_safe = True
        if code is not None:
            # Run check code
            is_code_not_safe = analyze_security(code)
            # Check if imports are missing
            if not is_code_not_safe:
                code = remove_entrypoint(code)
                # Check if imports are missing
                code_checked = self.missing_imports_chain.run(code=code)
                code_checked = None if code_checked == "None" else code_checked
                if code_checked is not None:
                    code = code_checked

        output: Dict[str, Any] = {self.output_key[0]: code, self.output_key[1]: expl}
        if self.return_source_documents:
            output["source_documents"] = docs
        if self.return_generated_question:
            output["generated_question"] = new_request
        if self.return_revision_request:
            output["revision_request"] = is_code_not_safe
        return output

    def save(self, file_path: Union[Path, str]) -> None:
        if self.get_chat_history:
            raise ValueError("Chain not savable when `get_chat_history` is not None.")
        super().save(file_path)


class ConversationalRetrievalCodeChain(BaseConversationalRetrievalCodeChain):
    """Chain for chatting with an index."""

    retriever: BaseRetriever
    """Index to connect to."""
    max_tokens_limit: Optional[int] = None
    """If set, restricts the docs to return from store based on tokens, enforced only
    for StuffDocumentChain"""

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.max_tokens_limit and isinstance(
            self.combine_docs_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_docs_chain.llm_chain.llm.get_num_tokens(doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        docs = self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        docs = await self.retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        return self._reduce_tokens_below_limit(docs)

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_CODE_PROMPT,
        chain_type: str = "stuff",
        verbose: bool = False,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        missing_imports_llm: Optional[BaseLanguageModel] = None,
        combine_docs_chain_kwargs: Optional[Dict] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseConversationalRetrievalCodeChain:
        """Load chain from LLM."""
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {"prompt": PROMPT}

        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            callbacks=callbacks,
            **combine_docs_chain_kwargs,
        )

        _llm = condense_question_llm or llm
        condense_question_chain = LLMChain(
            llm=_llm,
            prompt=condense_question_prompt,
            verbose=verbose,
            callbacks=callbacks,
        )

        _llm_3 = missing_imports_llm or llm
        missing_imports_chain = LLMChain(
            llm=_llm_3, prompt=prompt_missing_imports_check
        )

        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            missing_imports_chain=missing_imports_chain,
            callbacks=callbacks,
            **kwargs,
        )
