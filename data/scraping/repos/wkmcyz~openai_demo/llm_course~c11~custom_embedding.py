from util.proxy import set_proxy
import openai
import faiss
from llama_index import SimpleDirectoryReader, LangchainEmbedding, ServiceContext, GPTVectorStoreIndex
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser

if __name__ == '__main__':
    set_proxy()
    openai.api_key = ""

    # input
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)
    parser = SimpleNodeParser(text_splitter=text_splitter)
    documents = SimpleDirectoryReader('./data/faq/').load_data()
    nodes = parser.get_nodes_from_documents(documents)

    # embedding model
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ))
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    # dimension = 768
    # faiss_index = faiss.IndexFlatIP(dimension)
    # to index
    # index = GPTVectorStoreIndex(nodes=nodes, faiss_index=faiss_index, service_context=service_context)

    # to index
    index = GPTVectorStoreIndex(nodes=nodes, service_context=service_context)

    #
    query_engine = index.as_query_engine()
    # 我们是一家专注于提供优质服装和配饰的网上零售商。
    s = query_engine.query("你们是做什么的？")
    print("s:")
    print(s)
