from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from kanao.modules.get_api_key import get_api_key

def process_txt(file_path):
    api_key = get_api_key()

    # Raise an error if API key is not provided
    if not api_key:
        raise ValueError('OpenAI API key is not provided in the configuration file.')

    # Load the plain text file using PlainTextLoader
    loader = TextLoader(file_path)
    documents = loader.load()

    # Initialize OpenAIEmbeddings for text embeddings
    embeddings = OpenAIEmbeddings()

    # Create a ConversationalRetrievalChain with ChatOpenAI language model
    # and plain text search retriever
    txt_search = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3),
        retriever=txt_search.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
    )

    return chain