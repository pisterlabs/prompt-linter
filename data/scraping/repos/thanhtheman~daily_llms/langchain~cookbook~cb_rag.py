from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

text = """
    Tony lives in Toronto, Canada. He loves pets and he has 2 dogs name Fido and Rover. He always takes them for a walk everyday.
    One day, Tony was walking his dogs when he saw a cat. The cat was very cute and Tony wanted to pet it. But the cat ran away.
    Tony was sad and he went home. He told his wife, Jennie, about the cat and she said that they should get a cat.
"""
template = "Given the context below: {context}, please answer the following question: {question} in {language}"

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
prompt = ChatPromptTemplate.from_template(template)

vector_store = FAISS.from_texts([text], embedding=OpenAIEmbeddings())
retriever = vector_store.as_retriever()

chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | model 
    | StrOutputParser()
)
response = chain.invoke("What did the cat do?")

chain = {
    "context": itemgetter("question") | retriever, 
    "question": itemgetter("question"), 
    "language": itemgetter("language")
} | prompt | model | StrOutputParser()

response = chain.invoke({"question": "wwhat did tony do?", "language": "italian"})

print(response)
