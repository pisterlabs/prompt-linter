import logging
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import OpenAI

from assistant.semantic_search import SemanticSearch
from assistant import config
from assistant import prompts

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PersonalAssistant():
    """Chatbot assistant."""
    def __init__(self, **kwargs) -> None:
        openai_api_key = os.environ["OPENAI_API_KEY"]
        self.llm = OpenAI(temperature=config.temperature, 
                          model_name=config.model_name, max_tokens=config.max_tokens, 
                          openai_api_key=openai_api_key, **kwargs)

    def answer(self, query: str) -> str:
        """Answer the question."""
        LOGGER.info(f"Enter answer module with the following query: {query}")
        semanticsearch = SemanticSearch()
        contexts = semanticsearch.search(query=query)
        prompt = PromptTemplate(template=prompts.prompt_template, input_variables=["context", "query"])
        llmchain = LLMChain(prompt=prompt, llm=self.llm)
        context = contexts[0].page_content # For now, we consider only one context
        LOGGER.info(f"The context used to answer is: {context}")
        answer = llmchain.predict(context=context, query=query) 
        LOGGER.info(f"Exit answer module with the following answer: {answer}")
        return answer


pa = PersonalAssistant()
pa.answer(query="What was his last experience?")