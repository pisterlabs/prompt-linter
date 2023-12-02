import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def main():
    chat = ChatOpenAI(openai_api_key=api_key)
    message = "I want you to act as {comedian}. Tell me a joke."
    chat_prompt = ChatPromptTemplate.from_template(message)
    request = chat_prompt.format_prompt(comedian="Jerry Seinfeld")

    # print(request)
    print(request.input_variables)

    response = chat(request.to_messages())
    # print(response)
    # print(type(response))
    print(response.content)


if __name__ == "__main__":
    main()
