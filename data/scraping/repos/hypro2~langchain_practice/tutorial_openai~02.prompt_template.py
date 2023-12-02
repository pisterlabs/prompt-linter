import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.prompts import ChatPromptTemplate, PromptTemplate

"""
프롬프트 템플릿을 정의하고 사용이 가능합니다. 
입력변수가 없는 프롬프트부터 여러개 입력변수를 사용하는 프롬프트까지 여러가지 사용할 수 있습니다.
템플릿을 정의하고 {}를 통해 variable이 되는 인자를 정해주면 format을 통해 계속 계속 사용 할 수 있습니다.
"""

# 입력 변수가 없는 프롬프트 예제
def no_input_prompt():
    no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")
    prompt = no_input_prompt.format()
    print(prompt)


# 하나의 입력 변수가 있는 예제 프롬프트
def one_input_prompt():
    one_input_prompt = PromptTemplate(template="Tell me a {adjective} joke.", input_variables=["adjective"],)
    prompt = one_input_prompt.format(adjective="funny")
    print(prompt)


# 여러 입력 변수가 있는 프롬프트 예제
def multiple_input_prompt():
    multiple_input_prompt = PromptTemplate(template="Tell me a {adjective} joke about {content}.",
                                           input_variables=["adjective", "content"],
                                           )
    prompt = multiple_input_prompt.format(adjective="funny", content="chickens")
    print(prompt)


# input_variables를 지정 안한 프롬프트 예제
def no_variable_prompt():
    no_variable_prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    prompt = no_variable_prompt.format(product="colorful socks")
    print(prompt)


# ChatPrompt 예제
def chat_prompt():
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{human_text}"),
    ])

    message = chat_prompt.format_messages(input_language="English",
                                          output_language="French",
                                          human_text="I love programming.")
    print(message)

if __name__=="__main__":
    no_input_prompt()
    one_input_prompt()
    multiple_input_prompt()
    no_variable_prompt()
    chat_prompt()
