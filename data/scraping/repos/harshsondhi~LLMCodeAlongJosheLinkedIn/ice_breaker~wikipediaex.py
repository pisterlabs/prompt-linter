from langchain.document_loaders import WikipediaLoader
import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt
from langchain.output_parsers import OutputFixingParser

# output_parser = DatetimeOutputParser()
#
# misformatted = result.content
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.cache import InMemoryCache
from langchain import PromptTemplate
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.output_parsers import DatetimeOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


def answer_question_about(person_name, question):
    docs = WikipediaLoader(query=person_name, load_max_docs=1)
    context_text = docs.load()[0].page_content

    os.environ["OPENAI_API_KEY"] = "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"
    openai.api_key = "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"
    api_key = "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"
    llm = OpenAI()
    chat = ChatOpenAI(openai_api_key=api_key)

    human_prompt_template = (
        "Answer this question\n{question}, here is some extra context:\n{document}"
    )
    human_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)

    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    request = chat_prompt.format_prompt(
        question=question, document=context_text
    ).to_messages()
    result = chat(request)

    print(result.content)


answer_question_about("Claude Shannon", "When was he born?")
