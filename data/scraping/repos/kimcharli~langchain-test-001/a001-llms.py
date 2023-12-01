from langchain.llms import OpenAI
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

'''from https://python.langchain.com/docs/get_started/quickstart.html'''

def llms():

    llm = OpenAI(openai_api_key=openai.api_key, openai_organization=openai.organization)
    chat_model = ChatOpenAI(openai_api_key=openai.api_key, openai_organization=openai.organization)

    print(f"==\nllm.predict('Hi!'):\n  {llm.predict('Hi!')}")

    print(f"==\nchat_model.predict('Hi!'):\n{  chat_model.predict('Hi!')}")

    text = "What would be a good company name for a company that makes colorful socks?"
    print(f"==\ntest: {text}")

    print(f"==\nllm.predict(text):\n  {llm.predict(text)}")

    print(f"==\nchat_model.predict(text):\n  {chat_model.predict(text)}")


    text = "What would be a good company name for a company that makes colorful socks?"
    print(f"==\ntext: {text}")

    messages = [HumanMessage(content=text)]

    print(f"==\nllm.predict_messages(messages):  {llm.predict_messages(messages)}")

    print(f"==\nchat_model.predict_messages(messages):  {chat_model.predict_messages(messages)}")



## promtp templates
def promtp_templates():

    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    prompt.format(product="colorful socks")


    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")


## output parsers
def output_parsers():
    from langchain.schema import BaseOutputParser

    class CommaSeparatedListOutputParser(BaseOutputParser):
        """Parse the output of an LLM call to a comma-separated list."""


        def parse(self, text: str):
            """Parse the output of an LLM call."""
            return text.strip().split(", ")

    hi_bye = CommaSeparatedListOutputParser().parse("hi, bye")
    print(f"==\nhi_bye: {hi_bye}")


if __name__ == "__main__":
    import openai
    from util import parse_openai_api_key_file

    parse_openai_api_key_file()

    llms()
    promtp_templates()
    output_parsers()
