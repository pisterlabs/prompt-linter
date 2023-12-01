# from git import Repo
import os
from dotenv import load_dotenv
from langchain import hub # for default prompt
from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# for chats and memory
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain

# vector store
from langchain.vectorstores import Chroma, FAISS

# document loaders with a text splitter
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser

# Embedder
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings, GPT4AllEmbeddings, HuggingFaceInstructEmbeddings, OpenAIEmbeddings


# Loading env variables
load_dotenv()
repo_path = "/home/jovyan/LLmao/test_repo" # establish path to data files

n_gpu_layers = 51 # seems like only 51 layers for code Llama
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="/home/jovyan/codellama-34b-instruct.Q4_K_M.gguf",
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     callback_manager=callback_manager,
#     f16_kv=True,
#     verbose=True,  # Verbose is required to pass to the callback manager
#     temperature=0,
#     max_tokens=5000, # generated tokens
#     n_ctx=5000, # context length
# )

llm = LlamaCpp(
    model_path="/home/jovyan/llama-2-70b-chat.Q4_K_M.gguf",  # Make sure the model path is correct for your system!
    n_gpu_layers=70,
    n_batch=512,
    callback_manager=callback_manager,
    f16_kv=True,
    verbose=False,  # Verbose is required to pass to the callback manager
    temperature=0,
    max_tokens=5000, # generated tokens
    n_ctx=5000, # context length
)

print("finished setting up model")

# Loader and Splitter
loader = GenericLoader.from_filesystem(
    repo_path + "/libs/langchain/langchain",
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()
print(f"number of documents: {len(documents)} \n")

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
print(f"number of chunks of text: {len(texts)} \n")

# Embedding Model
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cuda"} # can change from cpu to cuda.
encode_kwargs = {"normalize_embeddings": True}
embedder_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# print(os.getenv("OPENAI_API_TYPE"), os.getenv("OPENAI_API_BASE"), os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_API_VERSION"))

# os.environ["OPENAI_API_TYPE"]="azure"
# os.environ["OPENAI_API_BASE"]=os.getenv("OPENAI_API_BASE")
# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_VERSION"]=os.getenv("OPENAI_API_VERSION")

# embedder_model = OpenAIEmbeddings(
#     openai_api_type="azure",
#     openai_api_base=os.getenv("OPENAI_API_BASE"),
#     model=os.getenv("OPENAI_API_ENGINE"),
#     deployment=os.getenv("OPENAI_API_ENGINE"),
# )

# vector store Chroma db
db = Chroma.from_documents(documents=texts, embedding=embedder_model)

retriever = db.as_retriever(
    search_type="mmr",  # "similarity" is another useful type as well
    search_kwargs={"k": 8}, # max marginal relevance set as 8 (ie. 8 most relevant)
)

print("Succesfully set up db and embedder!")

template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
Context: {context}
User: {question}
[/INST]"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# QA_chain_prompt = hub.pull("rlm/rag-prompt-llama")

question = "How can I initialize a reAct agent?"

# question2 = """[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# <</SYS>>
# Context: {context}
# User: How can I initialize a reAct agent?
# [/INST]"""

# docs = retriever.get_relevant_documents(question)

# print("retrived docs\n")
# print(docs)
# print()

chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever, 
    chain_type_kwargs={"prompt":QA_CHAIN_PROMPT}
    )

result = chain({"query": question})
print(result)
