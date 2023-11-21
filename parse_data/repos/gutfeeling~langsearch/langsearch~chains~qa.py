import logging

from langchain.chains.base import Chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAIChat
from pydantic import Extra

from langsearch.pipelines.common.index import SimpleIndexPipeline
from langsearch.utils import openai_length_function

logger = logging.getLogger(__name__)


class QAChain(Chain):
    class Config:
        extra = Extra.allow

    def __init__(self,
                 *args,
                 qa_chain=load_qa_chain(llm=OpenAIChat(temperature=0)),
                 qa_chain_question_input_name="question",
                 document_search_question_input_name="question",
                 length_function=openai_length_function,
                 max_context_size=4096,
                 method_or_docs=SimpleIndexPipeline().get_similar_sections,
                 method_args=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.qa_chain = qa_chain
        self.qa_chain_question_input_name = qa_chain_question_input_name
        self.document_search_question_input_name = document_search_question_input_name
        self.length_function = length_function
        self.max_context_size = max_context_size
        self.method_or_docs = method_or_docs
        if method_args is None:
            method_args = {"limit": 5}
        self.method_args = method_args

    @property
    def input_keys(self):
        return list({self.qa_chain_question_input_name, self.document_search_question_input_name})

    @property
    def output_keys(self):
        return self.qa_chain.output_keys + ["docs"]

    def _call(self, inputs):
        if callable(self.method_or_docs):
            docs = self.method_or_docs(inputs[self.document_search_question_input_name], **self.method_args)
        else:
            docs = self.method_or_docs
        if isinstance(self.qa_chain, StuffDocumentsChain):
            prompt_template = self.qa_chain.llm_chain.prompt
            empty_prompt = prompt_template.format(**{input_var: "" for input_var in prompt_template.input_variables})
            prompt_length = self.length_function(empty_prompt)
            question_length = self.length_function(inputs[self.qa_chain_question_input_name])
            # -256 minimum for answer, -100 for safety
            remaining = self.max_context_size - prompt_length - question_length - 256 - 100

            token_count = 0
            trimmed_docs = []
            for doc in docs:
                token_count += self.length_function(doc.page_content) + self.length_function("\n\n")
                if token_count > remaining:
                    break
                trimmed_docs.append(doc)
            docs = trimmed_docs
        # TODO: how to pass max_tokens in map_reduce_chain?
        qa_chain_inputs = inputs.copy()
        if self.document_search_question_input_name != self.qa_chain_question_input_name:
            del qa_chain_inputs[self.document_search_question_input_name]
        return {"docs": docs,
                **self.qa_chain({**qa_chain_inputs, self.qa_chain.input_key: docs}, return_only_outputs=True)
                }
