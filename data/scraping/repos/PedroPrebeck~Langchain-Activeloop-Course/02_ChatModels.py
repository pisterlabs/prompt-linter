from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(model="gpt-4", temperature=0.7)

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(content="Translate the following sentence: I love programming."),
]

print(chat(messages))

batch_messages = [
    [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="Translate the following sentence: I love programming."),
    ],
    [
        SystemMessage(
            content="You are a helpful assistant that translates French to English."
        ),
        HumanMessage(
            content="Translate the following sentence: J'aime la programmation."
        ),
    ],
]
print(chat.generate(batch_messages))
