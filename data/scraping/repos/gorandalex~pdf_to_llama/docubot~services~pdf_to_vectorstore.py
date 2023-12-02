import pickle

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
import openai
# from environs import Env

# env = Env()
# env.read_env('.env')
# openai.api_key = env.str('OPENAI_API_KEY')


async def pdf_to_vectorstore(file_path):
    # dir_path = Path.cwd()
    # path = str(Path(dir_path, 'pdf', 'CV - Junior Python Developer, Andrii Martyniuk.pdf'))
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents=pages)

    embeddings = OpenAIEmbeddings()
    # db_name = 'vector_db'
    db = FAISS.from_documents(chunks, embeddings)

    with open(f"{file_path[:-4]}.pkl", "wb") as f:
        pickle.dump(db, f)
    # try:
    #     vectorstore = FAISS.load_local(db_name, embeddings)
    # except Exception as e:
    #     print('Creating db....')
    #     vectorstore = FAISS.from_documents(chunks, embeddings)
    #     vectorstore.save_local(db_name)
    #     print('DB created')

    return db

