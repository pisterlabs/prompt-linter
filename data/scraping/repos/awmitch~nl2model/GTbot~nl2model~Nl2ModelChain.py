from langchain.chains import FlareChain
import logging
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains.flare.base import _low_confidence_spans
from typing import List, Tuple, Dict, Any,Optional
from .Nl2Modelica import ModelObject
from OMPython import ModelicaSystem
from io import StringIO
import sys
import re

# Custom Chain
class ModelicaFlareChain(FlareChain):
    modelica_model: ModelObject
    min_token_gap: int = 32#5
    num_pad_tokens: int = 16#2
    start_with_retrieval: bool = False
    def __init__(self, modelica_model, **kwargs):
        super().__init__(modelica_model=modelica_model, **kwargs)
        self.modelica_model = modelica_model
    def _do_generation(
        self,
        questions: List[str],
        user_input: str,
        response: str,
        _run_manager: CallbackManagerForChainRun,
    ) -> Tuple[str, bool]:
        callbacks = _run_manager.get_child()
        docs = []
        logging.info(f"flare Context: {self.modelica_model.modelica_context}")
        logging.info(f"flare modelica_model: {id(self.modelica_model)}")

        for question in questions:
            docs.extend(self.retriever.get_relevant_documents(question))
        if docs:
            self.modelica_model.modelica_context += "\n\n".join(d.page_content for d in docs)
        user_input += self.modelica_model.modelica_input
        logging.info(f"user_input:{self.modelica_model.modelica_input}")
        result = self.response_chain.predict(
            user_input=user_input,
            context=self.modelica_model.modelica_context,
            response=response,
            callbacks=callbacks,
        )
        marginal, finished = self.output_parser.parse(result)
        return marginal, finished
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        user_input = inputs[self.input_keys[0]]

        response = ""

        for i in range(self.max_iter):
            _run_manager.on_text(
                f"Current Response: {response}", color="blue", end="\n"
            )
            _input = {"user_input": user_input, "context": self.modelica_model.modelica_context, "response": response}
            tokens, log_probs = self.response_chain.generate_tokens_and_log_probs(
                _input, run_manager=_run_manager
            )
            low_confidence_spans = _low_confidence_spans(
                tokens,
                log_probs,
                self.min_prob,
                self.min_token_gap,
                self.num_pad_tokens,
            )
            initial_response = response.strip() + " " + "".join(tokens)
            if not low_confidence_spans:
                response = initial_response
                final_response, finished = self.output_parser.parse(response)
                if finished:
                    return {self.output_keys[0]: final_response}
                continue

            marginal, finished = self._do_retrieval(
                low_confidence_spans,
                _run_manager,
                user_input,
                response,
                initial_response,
            )
            response = response.strip() + marginal
            if finished:
                break
        return {self.output_keys[0]: response}

    
        