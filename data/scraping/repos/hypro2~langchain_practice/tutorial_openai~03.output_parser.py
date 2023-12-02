import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
from langchain.output_parsers import (PydanticOutputParser,
                                      OutputFixingParser,
                                      RetryWithErrorOutputParser,
                                      CommaSeparatedListOutputParser,
                                      )
from pydantic import BaseModel, Field, validator


"""
일반적으로 LLM은 텍스트를 출력합니다. 하지만 보다 구조화된 정보를 얻고 싶을 수 있습니다.
이런 경우 출력 파서를 이용하여 LLM 응답을 구조화할 수 있습니다.
출력 파서는 두 가지 컨셉을 갖고 있습니다.

!custom_parser로 구조 먼저보기!

- Format instructions : 원하는 결과의 포멧을 지정하여 LLM에 알려줍니다.
- Parser : 원하는 텍스트 출력 구조 (보통 json) 을 추출하도록 합니다.

이 출력 구문 분석기를 사용하면 사용자가 임의의 JSON 스키마를 지정하고 해당 스키마를 준수하는 JSON 출력에 대해 LLM을 쿼리할 수 있습니다.

대규모 언어 모델은 누수가 있는 추상화된 모델?이라 올바른 형식의 JSON을 생성할 수 있는 충분한 사이즈 갖춘 LLM을 사용해야 합니다. 

Pydantic을 사용하여 데이터 모델을 선언하세요. 
Pydantic의 BaseModel은 Python 데이터 클래스와 비슷하지만, "실제 유형 검사 + 강제성"이 있습니다.
"""


def comma_parser():
    """
    CommaSeparatedListOutputParser

    LLM에게 명시적으로 get_format_instructions을 통해서 원하는 결과의 포멧을 지정하여 LLM에 알려줍니다.
    아래 구문이 쿼리문으로 전달 됩니다. 그 만큼 토큰이 소비됩니다.
    "Your response should be a list of comma separated values, "
    "eg: `foo, bar, baz`"
    """

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template="List five {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions}
    )

    model = OpenAI(temperature=0, openai_api_key=openai_api_key)
    _input = prompt.format(subject="ice cream flavors") # 'List five ice cream flavors.\nYour response should be a list of comma separated values, eg: `foo, bar, baz`'
    output = model(_input)
    print(output_parser.parse(output)) # 리스트로 반환
    return output


class Joke(BaseModel):
    """
    JSON parser / Function calling과 유사하게 사용할 수 있습니다.

    Pydantic을 사용하면 사용자 지정 유효성 검사 로직을 쉽게 추가할 수 있습니다.
    Pydantic의 BaseModel은 Python 데이터 클래스와 비슷하지만, 실제 유형 검사 + 강제성이 있습니다.
    """
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # 유형 검사, 생성된  setup 필드에 ?표로 끝나는지 간단히 검사 가능하다.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


def json_parser():
    """
    LLM에게 명시적으로 get_format_instructions을 통해서 원하는 결과의 포멧을 지정하여 LLM에 알려줍니다.
    아래 구문이 쿼리문으로 전달 됩니다. 그 만큼 토큰이 소비됩니다.

    The output should be formatted as a JSON instance that conforms to the JSON schema below.

    As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
    the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

    Here is the output schema:
    ```
    {schema}
    ```
    """
    parser = PydanticOutputParser(pydantic_object=Joke)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = OpenAI(temperature=0,
                   callbacks=([StreamingStdOutCallbackHandler()]),
                   streaming=True ,
                   verbose=True,
                   openai_api_key=openai_api_key)

    prompt_and_model = prompt | model
    output = prompt_and_model.invoke({"query": "Tell me a joke."})

    print(output)
    return output


# fix retry parser


class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")


def fix_retry_parser():
    parser = PydanticOutputParser(pydantic_object=Action)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt_value = prompt.format_prompt(query="who is leo di caprios gf?")
    bad_response = '{"action": "search"}'
    # parser.parse(bad_response)

    """
    parser.parse(bad_response) 실행한다면 오류 action_input이 없기 때문에 에러가 발생한다. 
    
    langchain.schema.output_parser.OutputParserException: Failed to parse Action from completion {"action": "search"}. Got: 1 validation error for Action
    action_input
    field required (type=value_error.missing)
    
    에러가 발생하면 이런 프롬프트 템플릿 안에서 자동으로 실행됨.
    'Prompt:
    {prompt}
    Completion:
    {completion}
    
    Above, the Completion did not satisfy the constraints given in the Prompt.
    Details: {error}
    Please try again:'
    """

    model = OpenAI(temperature=0, openai_api_key=openai_api_key)

    # Auto-Fixing Parser 활용
    fix_parser = OutputFixingParser.from_llm(parser=parser, llm=model, max_retries=3)
    output = fix_parser.parse(bad_response)
    print(output)

    #대신, 프롬프트 (원래 출력뿐만 아니라)를 통과하는 RetryOutputParser를 사용하여 더 나은 응답을 얻기 위해 다시 시도 할 수 있습니다.
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model, max_retries=3)
    output = retry_parser.parse_with_prompt(bad_response, prompt_value)
    print(output)
    return



"""

Custom으로 parser를 정의하고 싶으면 BaseOutputParser를 상속받아서 정의하면 됩니다. 
간단히는 parser 함수만 새롭게 정의해서 사용하면 자신만의 커스텀 아웃풋 파서를 생성할 수 있습니다. 
그외에도 get_format_instructions를 정의해서 string 타입으로 return 한다면 프롬프트에 담아서 전달할 수 있습니다.

"""

class CustomSpaceSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(" ")


def custom_parser():
    parser = CustomSpaceSeparatedListOutputParser()

    prompt = PromptTemplate(
        template="Answer the user query.\n\n{query}\n",
        input_variables=["query"],
    )

    model = OpenAI(temperature=0,
                   callbacks=([StreamingStdOutCallbackHandler()]),
                   streaming=True ,
                   verbose=True,
                   openai_api_key=openai_api_key)

    prompt_and_model = prompt | model | parser

    output = prompt_and_model.invoke({"query": "Tell me a joke."})
    print(output)
    return output

if __name__=="__main__":
    comma_parser()
    json_parser()
    fix_retry_parser()
    custom_parser()
