import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def call_chat(formatted_prompt):
    chat = ChatOpenAI(openai_api_key=api_key)
    return chat(formatted_prompt.to_messages())


def main():
    system_template = (
        """You are a helpful AI assitant who speaks in the style of {style}."""
    )
    human_template = """I want you to provide travel recommendations for {place}"""
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Now, we need to compose them together to make a chatprompt.
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    request = chat_prompt.format_prompt(style="a pirate", place="New York City")
    print(request.to_messages())
    # to_messages formts it so it can be passed directly to the model as a python list of messages.
    result = call_chat(request)
    # print(result.content)


if __name__ == "__main__":
    main()
