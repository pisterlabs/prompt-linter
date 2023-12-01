from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ChatVectorDBChain
from langchain.llms import OpenAI
import os
import api_key as key

os.environ['OPENAI_API_KEY'] = key.OPENAI_API_KEY

embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory='./monai_gpt_db', embedding_function=embeddings)

chat_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name='gpt-4'), vectordb, return_source_documents=True)

query = "how does the Orientation function works in monai"
result = chat_qa({"question": query, "chat_history": ""})

print('Answer:')
print(result["answer"])