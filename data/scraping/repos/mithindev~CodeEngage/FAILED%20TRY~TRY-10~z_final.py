# SCRAPING
import os
import git
import shutil

# CONVERTING TO VECTOR
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

def clone_and_convert_to_text(repo_url, repo_dir, output_dir):
    # Remove the existing directory if it exists
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    # Clone the repository to the local directory
    repo = git.Repo.clone_from(repo_url, repo_dir)

    # Create the output directory for text files if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Delete existing text files in the output directory
        for file_name in os.listdir(output_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(output_dir, file_name)
                os.remove(file_path)

    # Walk through the local repository directory, read the content of each file,
    # and save it as a text file in the output directory
    for root, _, files in os.walk(repo_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Check if the file is a binary file (e.g., images, binaries) and skip it
            if os.path.isfile(file_path) and not file_name.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()

                # Save the content as a text file in the output directory with the same name
                txt_file_path = os.path.join(output_dir, file_name + ".txt")
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(content)

    # Clean up by removing the cloned repository
    shutil.rmtree(repo_dir)

def create_vector_db(data_path, db_faiss_path):
    loader = DirectoryLoader(data_path, glob='*.txt')

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='microsoft/unixcoder-base', model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(db_faiss_path)



if __name__ == "__main__":
    repo_url = "https://github.com/mithindev/mithindev"
    repo_dir = "cloned_repo"
    output_dir = "data"

    DATA_PATH = 'data/'
    DB_FAISS_PATH = 'vectorstore/db_faiss'

    create_vector_db(DATA_PATH, DB_FAISS_PATH)