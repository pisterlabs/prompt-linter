"""
使用 LangChain 进行文档问答
"""

# pip install chromadb
from langchain.document_loaders import DirectoryLoader

# 加载文件夹中的所有txt类型的文件，并转成 document 对象
loader = DirectoryLoader('./data/', glob='**/*.txt')
documents = loader.load()
# 接下来，我们将文档拆分成块。
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# 然后我们将选择我们想要使用的嵌入。
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
# 我们现在创建 vectorstore 用作索引，并进行持久化
from langchain.vectorstores import Chroma

# vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./vector_store")
# vector_store.persist()
vector_store = Chroma(persist_directory="./vector_store", embedding_function=embeddings)

from langchain.llms import OpenAI

llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    max_tokens=1024,
    verbose=True,
)
from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm=llm, chain_type="stuff")


# print(chain.input_keys, chain.output_keys)

def generate(query):
    docs = vector_store.similarity_search(query, k=4)
    # chain.run(input_documents=docs, question=query)
    res = chain(
        {
            "input_documents": docs,
            "question": query,
        },
        return_only_outputs=True,
    )["output_text"]
    print(res)


generate("出差申请单修改")
