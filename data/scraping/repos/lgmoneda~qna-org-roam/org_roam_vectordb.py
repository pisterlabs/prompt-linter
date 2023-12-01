from config import persist_directory, openai_api_key
from org_roam_parser import org_roam_df
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def org_roam_vectordb():
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma("langchain_store", embedding_function=embedding, persist_directory=persist_directory)
    roam_df = org_roam_df()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    for index, row in roam_df.iterrows():
        org_id = row["node_id"]
        title = row["node_title"]
        file_name = row["file_name"]
        node_hierarchy = row["node_hierarchy"]
        if type(row["node_text_nested_exclusive"]) == float:
            texts = [""]
        else:
            texts = text_splitter.split_text(row["node_text_nested_exclusive"])
        texts = ["[" + node_hierarchy + "] " + text for text in texts]
        metadatas = [
            {
                "source": f"{index}-{i}",
                "ID": org_id,
                "title": title,
                "hierarchy": node_hierarchy,
                "file_name": file_name,
            }
            for i in range(len(texts))
        ]
        ids = [f"{index}-{i}" for i in range(len(texts))]
        vectordb.add_texts(texts, metadatas=metadatas, ids=ids)
    vectordb.persist()
    print("VectorDB persisted successfully!")


org_roam_vectordb()
