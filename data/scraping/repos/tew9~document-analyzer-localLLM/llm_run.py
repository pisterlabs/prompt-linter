import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

openapi_key = os.environ.get("OPENAI_API_KEY")


def run_llm(file_path: str, prompt: str) -> str:
    """_summary_

    Args:
        file_path (str): file path to the text file containing the reviews
        prompt (str): prompt to be used to generate the insights
    Returns:
        str: insights from the reviews, e.g. the review that needs immediate attentions, and the ones with least impacts
    """

    # TODO: load the reviews from the file
    if not os.path.exists(file_path):
        raise FileNotFoundError("file does not exist")

    # TODO: load the reviews from the file
    loader = TextLoader(file_path)
    
    doc_loader = loader.load()

    # TODO: split the reviews into sentences
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=0
    )
    text_chunks = text_splitter.split_documents(documents=doc_loader)

    # TODO: load the embeddings
    embeddings = OpenAIEmbeddings()

    # TODO: vectorize the reviews
    print("vectorizing the reviews...")
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    # TODO: persist the vector store only once
    if not os.path.exists("docs/vector_store/"):
        os.mkdir("docs/vector_store")
        vector_store.save_local("docs/vector_store/review_index")
   
    # load the vector store
    print("loading local the vector store...")
    review_search = FAISS.load_local(
        "docs/vector_store/review_index", embeddings
    )

    # TODO: create retrieval chain
    qa =  RetrievalQA.from_chain_type(
        llm = OpenAI(), chain_type="stuff", retriever=review_search.as_retriever()
    )
    
    # TODO: generate insights using the prompt and the loaded reviews
    print("generating insights...")
    sys_context = "make sure you give full reviews, in a formatted order, remove the responses(like thanks..)."
    insights = qa.run(f'{prompt}. context: {sys_context}')

    return insights

