"""
similarity_search + 自定义 prompt 
"""

from langchain.document_loaders import DirectoryLoader
# pip install chromadb
from langchain.llms import OpenAI

# 加载文件夹中的所有txt类型的文件，并转成 document 对象
loader = DirectoryLoader('./data/', glob='**/*.txt')
documents = loader.load()
# 接下来，我们将文档拆分成块。
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# 然后我们将选择我们想要使用的嵌入。
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
# 我们现在创建 vectorstore 用作索引，并进行持久化
from langchain.vectorstores import Chroma

# vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./vector_store")
# vector_store.persist()
vector_store = Chroma(persist_directory="./vector_store", embedding_function=embeddings)
# 构建LLMChain
from langchain.prompts import PromptTemplate

prompt_template = """作为一个高精度的语义匹配模型，你的任务是根据给定的查询(query)判断主问题是否与之匹配。如果匹配，请输出1；否则请输出0。请注意不要包含任何额外的信息或解释。 请提供清晰明确的指导，以便用户了解所需完成的具体任务和如何满足这些需求。
    query: {query}
    primary questions: {context}
    匹配结果: """
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["query", "context"]
)
llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    max_tokens=1024,
    verbose=True,
)
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=PROMPT)
print(chain.input_keys, chain.output_keys)


def generate_blog_post(query):
    docs = vector_store.similarity_search(query, k=4)
    inputs = [{"query": query, "context": doc.page_content} for doc in docs]
    # print(chain.run(doc))
    results = chain.apply(inputs)
    for i, doc in enumerate(docs):
        print(results[i], [doc.page_content])
        print()


generate_blog_post("出差申请单修改")
