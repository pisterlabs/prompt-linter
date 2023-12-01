
#랭체인
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.prompt import Prompt
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


#FAISS ( vector db )
from langchain.vectorstores import FAISS

# 데이터프레임
import pandas as pd

# JSON으로 변환
import json



def namani(request_data):
    result_data = {"message": "좋은 하루 보내세요"}
    # tempurature : 0 ~ 1 로 높아질수록 랜덤한 답변 생성 / 창의력
    # llm = OpenAI(temperature=1)

    # usr_input = request_data["tempMessage"]
    # pre_input = request_data["preMessage"]
    usr_input = request_data.chatBotMessage["tempMessage"]
    pre_input = request_data.chatBotMessage["preMessage"]

    print(usr_input + " " + pre_input)

    # template
    template = "[prequestion]"+pre_input+"""
    
        [Instructions]
        - Act as a best friend
        - All answers should be between 10 and 35 characters long.
        - if you do not know answer, Refer to the {docs}
        - Your answer targets elderly people over 60 years old.

    """ 

    # amount_data = json.loads(request_data.json())["categoryData"]
    # amount_data_cols = amount_data.keys()  # 모든 key를 가져옴

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question IN KOREAN: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )


    # llm = ChatOpenAI(model_name="gpt-4-0613", temperature=1)
    # llm = OpenAI(model_name="gpt-4-0613", temperature=1)
    # llm = OpenAI(model_name="gpt-3.5-turbo-0613", temperature=1)
    # chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=1)


    # faiss를 통해 빠르게 참조 문서를 검색하고 연관된 답변을 제공할 수 있습니다
    # 단 LLM은 txt 기반이기 때문에 질문을 할 때는 다시 txt로 변환하여야 합니다

    # 1. 페이지가 많을 때 사용 (상품이 많다면, 백터를 통한 빠른 검색이 가능하지만 비용이 발생합니다)
    # query to llm(OpenAI)
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("chatbot/vector_faissdb", embeddings)

    docs = new_db.similarity_search(usr_input, k = 1)
    doc = " ".join([d.page_content for d in docs])

    # $$pay for OpenAI$$
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=1)
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(question = usr_input, docs = doc)
    print(result)
    # 문서 기반으로 질문
    result_data["message"] = result
    return result_data

if __name__=="__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("apikey")

    req = {
        "preMessage": "당신은 최근에 어떤 재미있는 이야기를 들었나요?",
        "tempMessage": "재미있는 이야기를 해주세요"
    }
    ans = (namani(req))

    print(ans.keys())
    # for i in ans.keys():
    #     print(ans[i])
    print(ans["message"])