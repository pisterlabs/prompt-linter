"""
使用向量数据库的例子
"""
import sys

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from langchain.document_loaders import TextLoader

from langchain.prompts import PromptTemplate

import env

# 切分文档
text_path= '../../texts/maodun.txt'
loader = TextLoader(text_path, encoding='utf8')

documents = loader.load()
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=20, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# 文档写入到向量数据库中。

from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever()

prompt_template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。

{context}

问: {question}
答:"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    # partial_variables={"language": "English"}
)
# print(prompt.format(question))
llm = OpenAI(max_tokens=8000,
             model_name="gpt-3.5-turbo-16k-0613")

chain_type_kwargs = {"prompt": prompt}
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    chain_type_kwargs=chain_type_kwargs,
                                    return_source_documents=True
                                    )
while True:
    print("=========================================")
    print("QUESION: ")
    question = sys.stdin.readline()
    result = chain({"query": question})
    print("ANSWER: ", result['result'])
    print("source_documents: ", result['source_documents'])
