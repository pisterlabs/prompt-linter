import os
from dotenv import load_dotenv

from langchain.output_parsers.enum import EnumOutputParser
from enum import Enum
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)
from langchain.chat_models import ChatOpenAI

load_dotenv()
model = ChatOpenAI(openai_api_key=api_key)


class Fruits(Enum):
    ORANGE = "orange"
    APPLE = "apple"
    BANANA = "banana"


def main():
    parser = EnumOutputParser(enum=Fruits)
    chat_messages = [
        SystemMessage(content="From the following message, extract the relevant fruit"),
        HumanMessage(content="My favorite fruit are apples"),
    ]
    output = model(chat_messages).content
    print(output)
    result = parser.parse(output)
    print(result)


if __name__ == "__main__":
    main()
