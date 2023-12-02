import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate

"""
답변 예시가 포함된 프롬프트 템플릿

"""

chat = ChatOpenAI(temperature=.7,
                  callbacks=([StreamingStdOutCallbackHandler()]),  # 콜백 기능 지원
                  streaming=True,
                  verbose=True,
                  openai_api_key=openai_api_key
                  )


def fewshot_prompt():
    # 답변 예시 준비
    examples = [
        {"input": "明るい", "output": "暗い"},
        {"input": "おもしろい", "output": "つまらない"},
    ]

    # 프롬프트 템플릿 생성
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="입력: {input}\n출력: {output}",
    )

    # 답변 예시를 포함한 프롬프트 템플릿 만들기
    prompt_from_string_examples = FewShotPromptTemplate(
        examples=examples,  # 답변 예시
        example_prompt=example_prompt,  # 프롬프트 템플릿
        prefix="모든 입력에 대한 반의어를 입력하세요",  # 접두사
        suffix="입력: {adjective}\n출력:",  # 접미사
        input_variables=["adjective"],  # 입력 변수
        example_separator="\n\n"  # 구분 기호
    )

    # 프롬프트 생성
    prompt = prompt_from_string_examples.format(adjective="큰")
    print(prompt)

    # 답변 생성
    response = chat([HumanMessage(content=prompt)])
    print(response)


def length_based_example_selector():
    # 답변 예시 준비
    examples = [
        {"input": "밝은", "output": "어두운"},
        {"input": "재미있는", "output": "지루한"},
        {"input": "활기찬", "output": "무기력한"},
        {"input": "높은", "output": "낮은"},
        {"input": "빠른", "output": "느린"},
    ]

    # 프롬프트 템플릿 생성
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="입력: {input}\n출력: {output}",
    )

    # LengthBasedExampleSelector 생성
    example_selector = LengthBasedExampleSelector(
        examples=examples,  # 답변 예시
        example_prompt=example_prompt,  # 프롬프트 템플릿
        max_length=10,  # 문자열의 최대 길이
    )

    # FewShotPromptTemplate 생성
    prompt_from_string_examples = FewShotPromptTemplate(
        example_selector=example_selector,  # ExampleSelector
        example_prompt=example_prompt,
        prefix="모든 입력에 대한 반의어를 입력하세요",
        suffix="입력: {adjective}\n출력:",
        input_variables=["adjective"],
        example_separator="\n\n"
    )

    # 프롬프트 생성
    prompt = prompt_from_string_examples.format(adjective="큰")
    print(prompt)

    # 답변 생성
    response = chat([HumanMessage(content=prompt)])
    print(response)


def semantic_similarity_example_selector():
    # 답변 예시 준비
    examples = [
        {"input": "밝은", "output": "어두운"},
        {"input": "재미있는", "output": "지루한"},
        {"input": "활기찬", "output": "무기력한"},
        {"input": "높은", "output": "낮은"},
        {"input": "빠른", "output": "느린"},
    ]

    # 프롬프트 템플릿 생성
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="입력: {input}\n출력: {output}",
    )

    # SemanticSimilarityExampleSelector 생성
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,  # 답변 예시
        embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key),  # 임베디드 생성 클래스
        vectorstore_cls=FAISS,  # 임베디드 유사 검색 클래스
        k=3  # 답변 예시 개수
    )

    # FewShotPromptTemplate 생성
    prompt_from_string_examples = FewShotPromptTemplate(
        example_selector=example_selector,  # ExampleSelector
        example_prompt=example_prompt,
        prefix="모든 입력에 대한 반의어를 입력하세요",
        suffix="입력: {adjective}\n출력:",
        input_variables=["adjective"],
        example_separator="\n\n"
    )

    # 프롬프트 생성
    prompt = prompt_from_string_examples.format(adjective="큰")
    print(prompt)

    # 답변 생성
    response = chat([HumanMessage(content=prompt)])
    print(response)


def MMR_example_selector():
    # 답변 예시 준비
    examples = [
        {"input": "밝은", "output": "어두운"},
        {"input": "재미있는", "output": "지루한"},
        {"input": "활기찬", "output": "무기력한"},
        {"input": "높은", "output": "낮은"},
        {"input": "빠른", "output": "느린"},
    ]

    # 프롬프트 템플릿 생성
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="입력: {input}\n출력: {output}",
    )

    # MaxMarginalRelevanceExampleSelector 생성
    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
        examples=examples,  # 답변 예시
        embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key),  # 임베디드 생성 클래스
        vectorstore_cls=FAISS,  # 임베디드 유사 검색 클래스
        k=3  # 답변 예시 개수
    )

    # FewShotPromptTemplate 준비
    prompt_from_string_examples = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="모든 입력에 대한 반의어를 입력하세요",
        suffix="입력: {adjective}\n출력:",
        input_variables=["adjective"],
        example_separator="\n\n"

    )

    # 프롬프트 생성
    prompt = prompt_from_string_examples.format(adjective="큰")
    print(prompt)

    # 답변 생성
    response = chat([HumanMessage(content=prompt)])
    print(response)


if __name__=="__main__":
    fewshot_prompt()
    length_based_example_selector()
    semantic_similarity_example_selector()
    MMR_example_selector()
    pass