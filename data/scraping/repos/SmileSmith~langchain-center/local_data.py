from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.chains import RetrievalQA
from dotenv.main import load_dotenv
import gradio as gr
import os
import ssl

# 解决NLTK下载报错问题
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 记录日志
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

# 参考https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/directory_loader.html 搭建webui，支持选项
openai_api_key = os.environ['OPENAI_API_KEY']
docs_dir = os.environ['DOC_DIRS']
persist_dir="stores"

def construct_vectorstore(docs_path, vectorstore_path):
    # 加载文件夹中的所有txt类型的文件
    loader = DirectoryLoader(docs_path, glob='**/*.md', loader_cls=UnstructuredMarkdownLoader)
    # 将数据转成 document 对象，每个文件会作为一个 document
    documents = loader.load()

    # 初始化加载器
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # 切割加载的 document
    split_docs = text_splitter.split_documents(documents)

    # 初始化 openai 的 embeddings 对象
    embeddings = OpenAIEmbeddings()
    # 将 document 通过 openai 的 embeddings 对象计算 embedding向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=vectorstore_path)
    # 持久化
    Chroma.persist(vectorstore)

    return vectorstore

# 将 document 通过 openai 的 embeddings 对象计算 embedding向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = construct_vectorstore(docs_dir, persist_dir)


def vearch(input_text):
    print("======================")
    print(input_text)
    print("----------------------")
    # 测试向量化匹配
    search_result = docsearch.similarity_search_with_score(input_text)
    print(search_result)
    print("----------------------")
    # 测试MMR
    retriever = docsearch.as_retriever(search_type="mmr")
    retrieve_result = retriever.get_relevant_documents(input_text)
    print(retrieve_result)
    print("======================")
    return search_result[0][0].page_content, retrieve_result[0].page_content

webui = gr.Interface(fn=vearch,
                     inputs=gr.components.Textbox(lines=7, label="输入您的文本"),
                     outputs=[gr.components.Text(label="相似度匹配结果"), gr.components.Text(label="MMR匹配结果")],
                     title="AI 向量化匹配")
webui.launch(share=True)

# 创建问答对象
# qa = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0, max_tokens=2048, openai_api_key=openai_api_key), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
# 进行问答
# result = qa({"query": query})
# print(result)