from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import dotenv

dotenv.load_dotenv()

PINECONE_API_KEY = dotenv.get_key(".env", "PINECONE_API_KEY")
OPENAI_API_KEY = dotenv.get_key(".env", "OPENAI_API_KEY")

query = "What is a team option and how does it work?"


def load_pdf():
    loader = UnstructuredPDFLoader("./docs/constitution.pdf")
    data = loader.load()
    return data


def process_text(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    return texts


def init_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")


def get_or_create_pinecone_data():
    index_name = "constitution"
    if index_name not in pinecone.list_indexes():
        data = load_pdf()
        texts = process_text(data)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        Pinecone.from_texts(
            [t.page_content for t in texts], embeddings, index_name=index_name
        )
    else:
        pass


def query_pinecone(query):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Pinecone.from_existing_index("constitution", embeddings)
    docs = docsearch.similarity_search(query)
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=query)


if __name__ == "__main__":
    init_pinecone()
    get_or_create_pinecone_data()
    result = query_pinecone(query)
