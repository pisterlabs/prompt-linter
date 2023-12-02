from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


if __name__ == "__main__":
    loader = TextLoader("E:\\VSCode\\vscode-python\\AI\\textcraft\\docs\\zelda.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()


    db = ElasticsearchStore.from_documents(
        docs,
        embeddings,
        es_url="http://124.220.51.225:9200/",
        index_name="test-basic",
    )

    db.client.indices.refresh(index="test-basic")

    query = "塞尔达有哪些主系列？"
    results = db.similarity_search(query)
    print(results)