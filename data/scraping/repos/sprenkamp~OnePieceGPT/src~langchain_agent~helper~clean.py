import os
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI


def clean_webtext_using_GPT(webtext: str, model_version: str = "gpt-3.5-turbo") -> str:
    llm = ChatOpenAI(
            temperature=0,  # Make as deterministic as possible
            model_name=model_version,
        )
    messages = [
    SystemMessage(
        content=f""" You are provided with raw text scraped from the web. Please clean the text removing any HTML tags, menu-buttons or other non-textual elements. You should only return information that is relevant to the web page. By doing so do not remove or change the meaning of the text. Retain the old text whenever possible. Also keep information like contact adresses, phone numbers, email addresses, etc. if they are present in the text. If none of the above is present in the text, please return "NO_INFORMATION".
        """
    ),
    HumanMessage(
        content=f"{webtext}"
    ),
    ]
    output = llm(messages)
    return output.content