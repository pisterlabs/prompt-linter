import docx
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from kanao.modules.get_api_key import get_api_key

def process_word_doc(file_path):
    api_key = get_api_key()

    # Raise an error if API key is not provided
    if not api_key:
        raise ValueError('OpenAI API key is not provided in the configuration file.')

    # Load the Word document using docx library
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    text = '\n'.join(full_text)

    # Initialize OpenAIEmbeddings for text embeddings
    embeddings = OpenAIEmbeddings()

    # Create a ConversationalRetrievalChain with ChatOpenAI language model
    # and plain text search retriever
    txt_search = Chroma.from_documents([text], embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3),
        retriever=txt_search.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
    )

    return chain