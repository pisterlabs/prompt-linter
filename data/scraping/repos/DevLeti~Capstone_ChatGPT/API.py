from typing import Annotated

from fastapi import FastAPI, Form

import dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import getArticle

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

def generateKeyword(user_keyword):
    keyword_gen_template = """아래의 질문을 몇가지 키워드로 요약해줘.
    질문: {question}"""

    prompt = PromptTemplate.from_template(keyword_gen_template)
    prompt.format(question="아이유 몇살이야?")

    api_key = os.environ["OPENAI_API_CD"]
    chat_model = ChatOpenAI(openai_api_key=api_key)
    keyword = chat_model.predict(prompt.format(question=user_keyword))
    print(keyword)
    return keyword

chat_model = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=os.environ["OPENAI_API_CD"])

template = ("당신은 주어진 articles를 기반으로 question을 답해야 합니다.\
            절대로 당신은 역질문을 해서는 안됩니다.\
            질문에 대한 답변만 하세요.\
            절대로 물음의 형태로 답변으로 내놓아서는 안됩니다.\
            당신은 질문을 할 수 없습니다.\
            기사의 내용 중 '예상', '전망'이란 단어가 들어간 문장은 유저가 '예상', '전망'에 대한 질문을 했을 때만 고려하세요.\
            유저가 이미 일어난 사실에 대한 정보를 원할 경우 기사의 내용 중 '예상', '전망'이란 단어가 들어간 문장은 고려하지마세요.\
            주어진 article로 기술된 정보를 통해 답할 수 있는 경우 답과 함께 근거 article를 붙여 대답하세요.\
            알 수 없는 경우 '모르겠습니다.'라고 답변하세요.")

human_template = "articles: {articles},\n" \
                  "question: {question_keyword}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chain = chat_prompt | chat_model

app = FastAPI()


@app.post("/search/")
async def search(data: Annotated[str, Form()]):
    # user_keyword = generateKeyword(data)
    article_string = getArticle.getArticleDetailBulkWithStr(data)
    result = chain.invoke({"articles": article_string[0:15000], "question_keyword": data})
    return {"result": result.content}

@app.post("/keyword/")
async def keyword(data: Annotated[str, Form()]):
    result = generateKeyword(data)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)