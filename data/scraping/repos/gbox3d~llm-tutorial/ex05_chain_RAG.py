#%%
import os
import time
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS

from transformers import AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from langchain import VectorDBQA, OpenAI

from dotenv import load_dotenv
import os
# .env 파일 로드
load_dotenv()
#%%
vectorstore = FAISS.from_texts(
    [
        "현재 대한민국의 페미니즘은 남성혐오와 여성우월주의의 혼합체이다. 타자성을 존중해달라는 사람들이 타자성은 지키지 않는다. 보수 진보를 막론하고 정권의 계속된 페미니즘 정책으로 지탄을 받으며, 아예 평등이 아닌 남성의 의무는 안 지면서 혜택을 요구하는 남성혐오, 여성우월주의의 온상으로 전락했다. 자신들이 그렇게도 혐오하던 여성혐오자들의 성향을 답습한 셈이다."
        "여성해방운동가들은 모두 브래지어를 태우고 남자를 증오하는 정신병자들입니다. 임신이 불행이고 아이를 낳는 것이 재앙이라고 생각하는 그런 정신나간 여자들을 어떻게 용인할 수 있겠습니까? -골다 메이어 (이스라엘의 여성 총리)",
        "2016년 이후 대한민국에서는 페미니즘을 긍정적으로 보는 시각과 부정적으로 보는 시각이 동시에 커짐으로서 양측 사이의 갈등이 빚어졌다.",
        "페미니즘은 여성의 권리를 위한 운동이다. 그러나 현대의 페미니즘은 여성의 권리를 위한 운동이 아니라 남성의 권리를 박탈하기 위한 운동이 되었다.",
        "당연하지만 페미니즘은 하나의 사상이자 이념(ideology)이기 때문에 논란과 문제점, 비판이 따라온다. 특히 현대에 주장하는 페미니즘은 남녀평등을 위한 여권 신장이 아니라, 여성들의 이익만을 대변하는 여성우월주의로 변질되었다는 비판적 주장이 그 예시. 한국 내부에는 1970년대 말에 들어오기 시작하여 한국의 여성에 대한 인식이 달라지는 계기가 되는 데 일조하였으며 1980년 중엽에 발전하였다. 이 당시의 여성인권 인식은 가정폭력에 보호받지 못할 정도로 기본적인 인권이 보장받지 못하던 시기였기에 양성평등을 위해서라도 여성권 신장이 필요하던 시기였다. 노동운동에도 저학력층 노동자에 여성노동자들이 같이 들어가 있던 만큼 노동운동권은 여성운동권과 연대을 할 수 밖에 없었으며, 이는 운동권이 현대의 변질된 여성운동에 적대적인 입장을 보이기 힘든 입장을 가지게 된 원인이기도 하다는 것을 감안하자면 그렇다. 하지만 이를 정치적인 목적으로 곡해하여 이용하는 여성인권단체 출신 정치인들과 어설프게 배운 페미니즘을 가부장적 사고와 연계시켜 레이디 퍼스트 형식으로 전파하는 사람들이 나타나게 되어, 무조건 여성의 이익만을 주장하는 쪽으로 발전하였다."
     ], 
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

#%% load openai model
chatModel = ChatOpenAI()
print(f'model ready: {chatModel}')

#%%
start_tick = time.time()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chatModel
    | StrOutputParser()
)
answer = chain.invoke("현대의 페미니즘은 무엇인가?")
print(answer)
print(f'Query time: {time.time() - start_tick}')

#%%
start_tick = time.time()
template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | chatModel
    | StrOutputParser()
)

answer = chain.invoke(
    {"question": "현대의 페미니즘은 무엇인가?", 
     "language": "korean"})
print(answer)
print(f'Query time: {time.time() - start_tick}')

# %%
