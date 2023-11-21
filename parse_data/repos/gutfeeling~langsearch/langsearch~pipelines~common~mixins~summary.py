import copy
import logging
import os

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.loading import load_llm_from_config
from langchain.prompts import PromptTemplate


from langsearch.exceptions import SettingsError
from langsearch.utils import openai_length_function


logger = logging.getLogger(__name__)


PROMPT = """{text}

Tl;dr
"""


class RecursiveReduceSummaryMixin:
    SUMMARY_SERIALIZED_LLM = {
        "_type": "openai",
        "temperature": 0,
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 512  # This should match SUMMARY_MAX_LENGTH
    }
    SUMMARY_LLM_MAX_CONTEXT_SIZE = 4096
    SUMMARY_LLM_LENGTH_FUNCTION = openai_length_function
    SUMMARY_MAX_LENGTH = 512
    SUMMARY_PROMPT = PROMPT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Use from_crawler() for these settings
        summary_serialized_llm = self.__class__.get_setting_from_partial_key(os.environ, "SUMMARY_SERIALIZED_LLM")
        if isinstance(summary_serialized_llm, str):
            summary_serialized_llm = self.load_params_from_file(summary_serialized_llm)
        # load_llm_from_config() modifies the dict, so we need to make a copy
        self.summary_llm = load_llm_from_config(copy.copy(summary_serialized_llm))
        summary_llm_max_context_size = self.__class__.get_setting_from_partial_key(
            os.environ, "SUMMARY_LLM_MAX_CONTEXT_SIZE"
        )
        if isinstance(summary_llm_max_context_size, str):
            try:
                summary_llm_max_context_size = int(summary_llm_max_context_size)
            except ValueError:
                raise SettingsError(
                    f"setting with partial key SUMMARY_LLM_CONTEXT_SIZE of class {self.__class__}"
                    f"must be convertible to int, but got '{summary_llm_max_context_size}'"
                )
        self.summary_llm_max_context_size = summary_llm_max_context_size
        summary_max_length = self.__class__.get_setting_from_partial_key(os.environ, "SUMMARY_MAX_LENGTH")
        if isinstance(summary_max_length, str):
            try:
                summary_max_length = int(summary_max_length)
            except ValueError:
                raise SettingsError(
                    f"setting with partial key SUMMARY_MAX_LENGTH of class {self.__class__}"
                    f"must be convertible to int, but got '{summary_max_length}'"
                )
        self.summary_max_length = summary_max_length
        summary_llm_length_function = self.__class__.get_setting_from_partial_key(
            os.environ, "SUMMARY_LLM_LENGTH_FUNCTION"
        )
        if isinstance(summary_llm_length_function, str):
            summary_llm_length_function = self.get_from_dotted(summary_llm_length_function)
        self.summary_llm_length_function = summary_llm_length_function
        summary_prompt = self.__class__.get_setting_from_partial_key(os.environ, "SUMMARY_PROMPT")
        summary_prompt_template = PromptTemplate(template=summary_prompt, input_variables=["text"])
        summary_prompt_length = self.summary_llm_length_function(
            summary_prompt_template.format(**{input_var: "" for input_var in summary_prompt_template.input_variables})
        )
        # - 50 for safety
        self.summarize_chain_max_stuff_tokens = (
                self.summary_llm_max_context_size - summary_prompt_length - self.summary_max_length - 50
        )
        self.summarize_chain = load_summarize_chain(
            llm=self.summary_llm,
            prompt=summary_prompt_template,
            chain_type="stuff"
        )
        # "\n\n" is used for joining docs in stuff document chain
        self.join_length = self.summary_llm_length_function("\n\n")

    def total_length(self, sections):
        return (
            sum([self.summary_llm_length_function(section) for section in sections]) +
            self.join_length * (len(sections) - 1)
        )

    def summarize(self, sections):
        while self.total_length(sections) > self.summary_max_length:
            summaries = []
            docs = []
            count = 0
            for section in sections:
                section_length = self.summary_llm_length_function(section)
                count += section_length + self.join_length
                if count > self.summarize_chain_max_stuff_tokens:
                    summary = self.summarize_chain.run(docs).strip("\n")
                    summaries.append(summary)
                    count = section_length
                    docs = [Document(page_content=section)]
                else:
                    docs.append(Document(page_content=section))
            summary = self.summarize_chain.run(docs).strip("\n")
            summaries.append(summary)
            sections = summaries
        return "\n\n".join(sections)
