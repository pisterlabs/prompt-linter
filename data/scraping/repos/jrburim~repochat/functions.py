import os
import requests
import io
from zipfile import ZipFile
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from deeplake.core.dataset import Dataset

EXTENSOES_DEV = ["py", "js", "ts", "html", "css", "scss", "json", "xml", "yml", "md", 
            "java", "cpp", "h", "c", "php", "rb", "go", "swift", "kt", "sql",
            "cs", "sh", "pyc", "rs", "tsx", "jsx", "sass", "less", "vue", "rbw",
            "pl", "ps1", "bat", "cmd"]

TMP_DIR = "tmp"

def download_and_extract_repo(url):
    # cria um diretorio tmp se não existir    
    os.makedirs(TMP_DIR, exist_ok=True)

    # Extrai o nome do repositório da URL
    repo_name = url.split("/")[-1].replace(".git", "")
    
    # Faz o download do arquivo zip do repositório
    zip_url = f"{url}/archive/refs/heads/main.zip"
    response = requests.get(zip_url)
    
    # Cria uma pasta com o nome do repositório e extrai o conteúdo do zip nela
    destination_folder = os.path.join(TMP_DIR, repo_name)
    os.makedirs(destination_folder, exist_ok=True)
    with ZipFile(io.BytesIO(response.content)) as zip_file:
        zip_file.extractall(destination_folder)
    
    extracted_dir: None;

    # le dentro da pasta do repositório, não entrar em sub pastas
    for dir_branch in os.listdir(destination_folder):
        if os.path.isdir(os.path.join(destination_folder, dir_branch)):
            extracted_dir = os.path.join(destination_folder, dir_branch)
            break

    if extracted_dir != None:
        # move o conteúdo da pasta extraída para a pasta do repositório
        for dir in os.listdir(extracted_dir):
            os.rename(os.path.join(extracted_dir, dir), os.path.join(destination_folder, dir))
        # deleta a pasta extraída
        os.rmdir(extracted_dir)

    return repo_name, destination_folder

def db_add_repo_files(db, repoName, repoFolder, extensoes_dev=EXTENSOES_DEV) -> Dataset:
    """
    Walk our target codebase and load all our files
    for chunking and then text embedding
    """    
    documents = []
    for dirpath, dirnames, filenames in os.walk(repoFolder):
        for file in filenames:
            if extensoes_dev is None or any(fnmatch.fnmatch(file, '*.' + ext) for ext in extensoes_dev):
                try: 
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    documents.extend(loader.load_and_split())
                except UnicodeDecodeError:
                    loader = TextLoader(os.path.join(dirpath, file), encoding='ISO-8859-1')
                    documents.extend(loader.load_and_split())    
                except Exception as e: 
                    print(e)

    for doc in documents:
      doc.metadata['repo'] = repoName
    
    # chunk our files
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    # generate text embeddings for our target codebase
    db.add_documents(chunks)
    return db

def get_retriever(db, repo):
    return db.as_retriever( search_kwargs={'distance_metric': 'cos', 'fetch_k': 100, 'maximal_marginal_relevance': True, 'k': 10, 'filter': {'metadata': { 'repo':repo }}})

def check_repo_in_db(db, repo):
    return len(db.vectorstore.search(filter={'metadata': { 'repo':repo }})['id']) > 0

import tiktoken

def calcular_total_tokens(nome_arquivo):
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
            conteudo = arquivo.read()
    except UnicodeDecodeError:
        with open(nome_arquivo, 'r', encoding='ISO-8859-1') as arquivo:  # Tente uma codificação diferente
            conteudo = arquivo.read()
    tokens = list(encoding.encode(conteudo, disallowed_special=()))
    total_tokens = len(tokens)
    return total_tokens

#total = calcular_total_tokens('meu_arquivo.txt')

import os
import fnmatch

def calcular_total_tokens_diretorio(diretorio, extensoes_dev=None):   
    total_tokens = 0
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            if extensoes_dev is None or any(fnmatch.fnmatch(file, '*.' + ext) for ext in extensoes_dev):
                total_tokens += calcular_total_tokens(os.path.join(root, file))
    return total_tokens

def custo(total_tokens):
    #$0.0001 / 1K tokens
    return (total_tokens / 1000) * 0.0001
    
def custo_embeddings_repo(diretorio):
    total_tokens = calcular_total_tokens_diretorio(diretorio, extensoes_dev=EXTENSOES_DEV)
    custoUSD = custo(total_tokens)
    return total_tokens, custoUSD


