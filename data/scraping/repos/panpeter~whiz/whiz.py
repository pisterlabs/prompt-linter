import os
import git
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Set OpenAI API key
# os.environ['OPENAI_API_KEY'] = "your-openai-api-key"

root_dir = './'
persist_directory = "./.chroma"
embeddings = OpenAIEmbeddings()

def is_file_ignored(file_path):
    git_repo = git.Repo(search_parent_directories=True)
    if not git_repo:
        return False
    try:
        gitignore = git_repo.git.check_ignore(file_path)
        return gitignore != ""
    except git.exc.GitCommandError:
        return False

if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print("Caching the project data...")
    docs = []
    file_extensions = ['.py', '.swift', '.kt', '.html', '.css', '.js', '.json', '.xml', '.java', '.php', '.go', '.rb', '.c', '.cpp', '.h', '.m', '.mm', '.dart', '.cs', '.ts', '.tsx', '.jsx', '.pyi', '.rs', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.awk', '.yml', '.yaml', '.toml', '.ini', '.cfg', '.conf', '.txt', '.md', '.rst', '.csv', '.tsv', '.sql', '.graphql', '.gql']
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            file_extension = os.path.splitext(file)[1]
            if file_extension in file_extensions and not is_file_ignored(os.path.join(dirpath, file)):
                print(f"Adding {file}...")
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
    print(f"{len(docs)} files...")

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    print(f"{len(texts)} texts...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()

print("Starting...")

retriever = db.as_retriever()
retriever.search_kwargs = {
    'distance_metric': 'cos',
    'fetch_k': 100,
    'maximal_marginal_relevance': True,
    'k': 10
}

model = ChatOpenAI(model='gpt-4')
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever, max_tokens_limit=4000)

chat_history = []

def fetch_result(question):
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    return result['answer']

while True:
    question = input("Q: ").strip()
    if question.lower() == "exit":
        break

    answer = fetch_result(question)
    print(f"{answer} \n")
