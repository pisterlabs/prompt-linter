
import os
import subprocess
import time

from git import Repo

from langchain.document_loaders import GoogleDriveLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import GitLoader, PyPDFLoader, PyPDFDirectoryLoader

API_KEY = os.environ.get('OPENAI_API_KEY')

# fetch from google drive
# full url: https://drive.google.com/drive/u/0/folders/1t41C3KyE6AxIBGg_B_ripKybbcACEDU3
# GOOGLE_DRIVE_FOLDER_ID = "1t41C3KyE6AxIBGg_B_ripKybbcACEDU3"
# loader = GoogleDriveLoader(folder_id=GOOGLE_DRIVE_FOLDER_ID, file_types=["pdf"])

REPO_LOCAL_PATH = "/tmp/lab-langchain-getting-started"

# fetch from github
if not os.path.exists(REPO_LOCAL_PATH):
    print("CLONING REPO")
    subprocess.run("rm -rf " + REPO_LOCAL_PATH, shell=True)
    Repo.clone_from("https://github.com/samber/lab-langchain-getting-started", to_path=REPO_LOCAL_PATH)
    print("CLONED REPO")

# load files
# loader = GitLoader(repo_path=REPO_LOCAL_PATH, branch=repo.head.reference, file_filter=lambda file_path: file_path.find("ar6/spm") != -1 and file_path.endswith(".pdf"))
print("LOADING PDF")
loader = PyPDFDirectoryLoader(
    path=REPO_LOCAL_PATH,
    glob="assets/ar6/full-report/*.pdf",
    # glob="assets/ar6/spm/*.pdf",
    recursive=True,
)
docs = loader.load()
print("LOADED PAGES:", len(docs))

# tokenize
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print("CHUNKS (size=1000):", len(chunks))

# prepare the embedding engine
# embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    # model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./transformers/all-MiniLM-L6-v2",
)

# setup in-memory vector store and run embedding against the chunks
print("GENERATING EMBEDDINGS")
start = time.time()
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
db.persist()
print("EMBEDDINGS OK\nThis took %.2f seconds" % (time.time()-start))

# convert vector store as retriever
retriever = db.as_retriever()

# build chain
llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY, model_name="gpt-3.5-turbo-16k")
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, verbose=True)

# cite sources
def process_llm_response(llm_response):
    return {
        "result": llm_response['result'],
        "sources": ["{} (page {})".format(source.metadata['source'], source.metadata['page']) for source in llm_response["source_documents"]],
    }

def ask(prompt):
    llm_response = qa_chain(prompt)
    return process_llm_response(llm_response)

def get_prompt():
    print("Type 'exit' to quit")

    while True:
        prompt = input("Enter a prompt: ")

        if prompt.lower() == 'exit':
            print('Exiting...')
            break
        else:
            try:
                result = ask(prompt)
                print("Result:\n", result['result'])
                print("\n")
                print("Sources:")
                for source in result['sources']:
                    print(" -", source)
                print("\n")
            except Exception as e:
                print(e)

if __name__ == "__main__":
    get_prompt()
