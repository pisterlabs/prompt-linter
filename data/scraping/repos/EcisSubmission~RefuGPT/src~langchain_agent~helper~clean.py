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
        content=f"""You are provided with raw content extracted from a web source. Your task is to clean and refine this content according to the following guidelines, the content will be used for a information retrieval database so remove any content that is not relevant for information retrieval.:
Retain Context-rich Content: Preserve paragraphs, detailed explanations, or other content that provides a complete context or full-fledged information. Remove any lines that are just keywords and don't provide any context, typically these are buttons on the website, but here they will be just text.
Default Response: If after processing, no relevant content remains or if the original text had no meaningful information, return "NO_INFORMATION". 
        """
    ),
    HumanMessage(
        content=f"{webtext}"
    ),
    ]
    output = llm(messages)
    return output.content