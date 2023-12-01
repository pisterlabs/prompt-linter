import os
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import re
import streamlit as st


api_keys = st.secrets["API_KEYS"]
MONGO_URL_findOne = st.secrets["MONGO_URL_findOne"]
MONGO_URL_updateOne = st.secrets["MONGO_URL_updateOne"]
MONGO_KEY = st.secrets["MONGO_KEY"]

def process_file(uploaded_file):
    st.spinner("读取文件中...")
    # Create a temporary file to store the uploaded file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()

    loader = PyPDFLoader(temp_file.name)
    documents = loader.load()
    file_name = re.sub('[^a-zA-Z0-9]', '', uploaded_file.name)[:63]

    os.unlink(temp_file.name)
    print(type(documents))
    print("这个地方是documents")
    print(len(documents))
    print(documents[0])

    # 进行文本分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    print(type(split_docs))
    print("这个地方是split_docs")
    print(len(split_docs))
    print(split_docs[0])

    return split_docs, file_name

st.title("📄 你可以上传PDF后再输入你的摘要，我可以帮你找到相关的来源")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    documents, file_name = process_file(uploaded_file)
    # Load embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=api_keys)
    pdfsearch = Chroma.from_documents(documents, embeddings)
    retriever = pdfsearch.as_retriever(search_kwargs={"k": 15})
    st.success("文件已读取")

# def render_file(file):
#     doc = fitz.open(stream=file)
#     print(N)
#     print(N[-1])
#     page = doc[N[-1]]
#     # Render the page as a PNG image with a resolution of 300 DPI
#     pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
#     image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
#     return image

def main():
    results_per_page = 10
    query = st.text_input("请输入摘要的自然段部分：")
    page = st.number_input("Page", min_value=1, max_value=300 // results_per_page + 1, value=1, step=1)

    if query:
        data = retriever.get_relevant_documents(query)
        num_pages = len(data) // results_per_page + (1 if len(data) % results_per_page > 0 else 0)
        start = (page - 1) * results_per_page
        end = page * results_per_page

        for match in data[start:end]:
            # print(match)
            with st.expander(f'**{match.metadata["page"]}**'):
                # lines = match.page_content.split("\n")
                # for line in lines:
                #     key, value = line.split(":", 1)
                st.info("下面是语义搜索到的正文：（可以复制至pdf中进行搜索与匹配）")
                st.markdown(match.page_content)
        st.write(f"Page {page} of {num_pages}")

if "shared" not in st.session_state:
    st.error("请在Home页输入邮箱")
elif st.session_state.shared:
    main()
else:
    st.error("邮箱未激活或已过期")