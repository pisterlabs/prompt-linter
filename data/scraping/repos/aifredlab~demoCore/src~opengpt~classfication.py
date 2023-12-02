import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    #SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import SystemMessage


def main(prompt):
    
    # FIXME :: DB
    template = """
        Here are the requirements
        1. 질의 내용에 대한 카테고리 분류작업
        2. 하기 카테고리중 1개의 결과만 리턴
        '보험료계산'
        '약관조회'
        '기타'
        3. 아래 json 양식으로 출력
        {"category" : ""}
    """
    #system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "질의 : {text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([SystemMessage(content=template), human_message_prompt])
    chain = LLMChain(
        llm=ChatOpenAI(),
        prompt=chat_prompt
    )
    jsonStr = chain.run(text=prompt)
    print(jsonStr)
    result = json.loads(jsonStr)

    return result
