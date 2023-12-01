from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.faiss import FAISS

load_dotenv()

texts = [
    "私の趣味は読書です。",
    "私の好きな食べ物はカレーです。",
    "私の嫌いな食べ物は饅頭です。",
]
vectorstore = FAISS.from_texts(texts, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

prompt = ChatPromptTemplate.from_template(
    """以下のcontextだけに基づいて回答してください。

{context}

質問: {question}
"""
)

model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

output_parser = StrOutputParser()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

result = chain.invoke("私の好きな食べ物はなんでしょう？")
print(result)
