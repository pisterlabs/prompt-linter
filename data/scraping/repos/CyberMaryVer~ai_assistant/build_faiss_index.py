import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from fastapi_app.chatbot.secret import OPENAI_API_KEY


def get_data(csv):
    data = pd.read_csv(csv, index_col=0)
    return data


def build_index_from_data(data):
    sources = [Document(page_content=' '.join(row.tolist()[:2]),
                        metadata={"source": row.tolist()[2]}) for i, row in
               data.iterrows()]
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=428, chunk_overlap=0)
    for source in sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    return search_index
