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

from feedback.product import products
#GPT
# from ML.openaikey import apikey

#키 등록
# import os
# os.environ["OPENAI_API_KEY"] = apikey

# 데이터프레임
import pandas as pd

# JSON으로 변환
import json

def mkreport(request_data):
    result_data = {"message": "월별 AI 분석"}
    # tempurature : 0 ~ 1 로 높아질수록 랜덤한 답변 생성 / 창의력
    # llm = OpenAI(temperature=1)

    # template
    template = """
        Act as a Financial Analyst

        [Instructions]
        "The input should be in the following format:
        'Category Name: Expense amount, Limit amount'
        Each English word corresponds to the mapped Korean term on the right."

        food : 식비
        traffic : 교통
        online : 온라인 쇼핑
        offline : 오프라인 쇼핑
        cafe : 카페/간식
        housing : 주거/통신
        fashion : 패션/미용
        culture : 문화/여가

        Based on the given expense and limit amounts for each category, please write compact feedback in Korean, within approximately 150 characters including spaces.
        and Based on the {question} of the respective user, refer to the {docs} and recommend one appropriate card and the reason for it.
    """

    request_data = json.loads(request_data.json())["categoryData"]
    request_data_cols = request_data.keys()  # 모든 key를 가져옴

    # 중간 txt
    consume_list = []
    for col_name in request_data_cols:
        consume_list += str(col_name) + ":"
        for amount in request_data[col_name]:
            consume_list += str(amount) + ","
        consume_list += "\n"
    query = ''.join(a for a in consume_list)

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=1)

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
    # llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=1)


    # faiss를 통해 빠르게 참조 문서를 검색하고 연관된 답변을 제공할 수 있습니다
    # 단 LLM은 txt 기반이기 때문에 질문을 할 때는 다시 txt로 변환하여야 합니다

    # 1. 페이지가 많을 때 사용 (상품이 많다면, 백터를 통한 빠른 검색이 가능하지만 비용이 발생합니다)
    # query to llm(OpenAI)
    # embeddings = OpenAIEmbeddings()
    # new_db = FAISS.load_local("vector_faissdb", embeddings)
    #
    # docs = new_db.similarity_search(input_txt, k = 1)
    # doc = " ".join([d.page_content for d in docs])
    #
    # result = llm(question = input_txt, docs = doc)

    # 2. 페이지가 적을 때 사용
    doc = products
    # result = llm(question=input_txt, docs=doc, messages=["1","2"])
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(question=query, docs=doc)

    # 문서 기반으로 질문

    result_data["message"] = result
    return result_data

if __name__=="__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("apikey")

    req = {
        "food": [400, 10000],
        "traffic": [7000, 10000],
        "online": [0, 10000],
        "offline": [9000, 10000],
        "cafe": [0, 10000],
        "housing": [0, 10000],
        "fashion": [12000, 10000],
        "culture": [0, 10000]
    }
    ans = (mkreport(req))

    # print(ans.keys())
    # for i in ans.keys():
    #     print(ans[i])
    print(ans["message"])
