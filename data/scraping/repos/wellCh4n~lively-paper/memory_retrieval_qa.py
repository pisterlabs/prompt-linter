import inspect
from typing import Dict, Any, Optional, List

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.schema import Document


class MemoryRetrievalQA(ConversationalRetrievalChain):

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(question, inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(question, inputs)  # type: ignore[call-arg]
        custom_docs = inputs.pop('custom_docs')
        docs = docs + custom_docs
        new_inputs = inputs.copy()
        new_inputs["question"] = question
        new_inputs["chat_history"] = chat_history_str
        answer = self.combine_docs_chain.run(
            input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs
        )
        output: Dict[str, Any] = {self.output_key: answer}
        if self.return_source_documents:
            output["source_documents"] = docs
        return output
