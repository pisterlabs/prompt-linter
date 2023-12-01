import logging
import os
from typing import List, Any, Dict

from langchain import OpenAI, PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatAnthropic
from langchain.schema import Document


# os.environ["LANGCHAIN_TRACING"] = "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreadSummarizer:
    def __init__(self,
                 use_anthropic: bool = False,
                 ):

        prompt_template = """Write a thorough summary of the following:


        "{text}"


        SUMMARY:"""

        self.base_summary_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

        # self.base_summary_prompt = PromptTemplate(
        #     template=THREAD_SUMMARY_PROMPT_TEMPLATE,
        #     input_variables=["text"])
        #
        # self.refine_prompt = PromptTemplate(
        #     template=REFINE_THREAD_SUMMARY_PROMPT_TEMPLATE,
        #     input_variables=["existing_answer", "text"])

        if use_anthropic:
            if os.getenv("ANTHROPIC_API_KEY") is None:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            self.llm = ChatAnthropic(temperature=0,
                                     max_tokens_to_sample=1000,
                                     callbacks=[StreamingStdOutCallbackHandler()],
)
            self.llm_model = self.llm.model
            self.dollars_per_token = 0.00000163
        if not use_anthropic or self.llm is None:
            self.llm = OpenAI(temperature=0, max_tokens=1000)
            self.llm_model = self.llm.model_name
            self.dollars_per_token = 0.00002

        self.chain = load_summarize_chain(self.llm,
                                          chain_type="refine",
                                          verbose=True,
                                          question_prompt=self.base_summary_prompt,
                                          # refine_prompt=self.refine_prompt,
                                          return_refine_steps=True,
                                          )

    async def summarize(self, thread_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(f"Summarizing {len(thread_chunks)} chunks...")
        chunks_as_documents = [Document(page_content=chunk["text"]) for chunk in thread_chunks]
        summary = await self.chain.acall({"input_documents":chunks_as_documents}, return_only_outputs=True)
        return summary


