import logging

from langchain.chains.base import Chain
from langchain.chains.hyde.prompts import PROMPT_MAP
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAIChat
from pydantic import Extra

from langsearch.chains.qa import QAChain

logger = logging.getLogger(__name__)


class HYDEChain(Chain):
    class Config:
        extra = Extra.allow

    def __init__(self,
                 *args,
                 hyde_llm_chain=LLMChain(
                     llm=OpenAIChat(temperature=0.7),
                     prompt=PROMPT_MAP["web_search"]
                 ),
                 hyde_llm_chain_question_input_name="QUESTION",
                 langsearch_qa_chain=QAChain(
                     document_search_question_input_name="hyde_llm_output"
                 ),
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.hyde_llm_chain = hyde_llm_chain
        self.hyde_llm_chain_question_input_name = hyde_llm_chain_question_input_name
        self.langsearch_qa_chain = langsearch_qa_chain

    @property
    def input_keys(self):
        return self.hyde_llm_chain.input_keys

    @property
    def output_keys(self):
        return self.langsearch_qa_chain.output_keys

    def _call(self, inputs):
        outputs = self.hyde_llm_chain(inputs, return_only_outputs=True)
        logger.debug("Hyde LLM chain outputs: %s", outputs)
        return self.langsearch_qa_chain(
            {
                self.langsearch_qa_chain.document_search_question_input_name: outputs[self.hyde_llm_chain.output_key],
                self.langsearch_qa_chain.qa_chain_question_input_name: inputs[self.hyde_llm_chain_question_input_name]
            },
            return_only_outputs=True
        )
