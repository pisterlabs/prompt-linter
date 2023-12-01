from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import Field

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

from demo.synthesizers.json_synthesizer.json_synthesizer_prompt import CHAT_PROMPT

class JsonSynthesizerChain(Chain):
    llm_chain: LLMChain
    text_splitter: TextSplitter = Field(
        default=RecursiveCharacterTextSplitter(chunk_overlap=500)
    )
    input_key: str = "text"
    output_key: str = "inputs"
    k: Optional[int] = None

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> JsonSynthesizerChain:
        _prompt = prompt or CHAT_PROMPT
        chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=chain, **kwargs)

    @property
    def _chain_type(self) -> str:
        raise NotImplementedError

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List]:
        docs = self.text_splitter.create_documents([inputs[self.input_key]])
        results = self.llm_chain.generate(
            [{"text": d.page_content} for d in docs], run_manager=run_manager
        )
        qa = [json.loads(res[0].text) for res in results.generations]
        return {self.output_key: qa}