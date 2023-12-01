import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser

output_parser = DatetimeOutputParser()

misformatted = result.content
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

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
api_key = "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"
llm = OpenAI()
chat = ChatOpenAI(openai_api_key=api_key)

# from langchain.output_parsers import CommaSeparatedListOutputParser
# output_parser = CommaSeparatedListOutputParser()
# print(output_parser.get_format_instructions())
#
# human_template = "{request}\n{format_instruction}"
# human_prompt = HumanMessagePromptTemplate.from_template(human_template)
#
# chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
# model_request = chat_prompt.format_prompt(request='write a poem about animals',format_instruction= output_parser.get_format_instructions()).to_messages()
#
# result = chat(model_request)
# print(result)


from langchain.output_parsers import DatetimeOutputParser

output_parser = DatetimeOutputParser()
print(output_parser.get_format_instructions())

template_text = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(template_text)

system_prompt = SystemMessagePromptTemplate.from_template(
    "You always reply to questions only in datetime patterns."
)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

# chat_prompt.format_prompt()

request = chat_prompt.format_prompt(
    request="What date was the 13th Amendment ratified in the US?",
    format_instructions=output_parser.get_format_instructions(),
).to_messages()

result = chat(request, temperature=0)
# print(result.content)

misformatted = result.content

new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=chat)
new_parser.parse(misformatted)
