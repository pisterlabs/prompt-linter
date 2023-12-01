import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
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

system_template = "You are a helpful assistant that translates complex legal terms into plain and understandable terms."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

legal_text = "The provisions herein shall be severable, and if any provision or portion thereof is deemed invalid, illegal, or unenforceable by a court of competent jurisdiction, the remaining provisions or portions thereof shall remain in full force and effect to the maximum extent permitted by law."
example_input_one = HumanMessagePromptTemplate.from_template(legal_text)

plain_text = "The rules in this agreement can be separated."
example_output_one = AIMessagePromptTemplate.from_template(plain_text)

human_template = "{legal_text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_input_one, example_output_one, human_message_prompt]
)

some_example_text = "The grantor, being the fee simple owner of the real property herein described, conveys and warrants to the grantee, his heirs and assigns, all of the grantor's right, title, and interest in and to the said property, subject to all existing encumbrances, liens, and easements, as recorded in the official records of the county, and any applicable covenants, conditions, and restrictions affecting the property, in consideration of the sum of [purchase price] paid by the grantee."
request = chat_prompt.format_prompt(legal_text=some_example_text).to_messages()

result = chat(request)
print(result.content)
