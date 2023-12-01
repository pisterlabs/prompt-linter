
import environment
import os

def cohereEmbeddings():
    from langchain.embeddings import CohereEmbeddings
    embedding = CohereEmbeddings(cohere_api_key=os.environ.get("COHERE_API_KEY"))
    return embedding

def openaiEmbeddings():
    from langchain.embeddings import OpenAIEmbeddings
    embedding = OpenAIEmbeddings()
    return embedding

def llamaCppEmbeddings():
    from langchain.embeddings import LlamaCppEmbeddings
    path = "../gpt4all/chat/ggml-gpt4all-l13b-snoozy.bin"
    embedding = LlamaCppEmbeddings(model_path=path)
    return embedding

def huggingFaceEmbedding():
    from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings()
    return embedding

def defaultEmbeddings():
    # embedding = huggingFaceEmbedding()
    # embedding = llamaCppEmbeddings()
    embedding = cohereEmbeddings()
    return embedding

defaultEmbeddings = defaultEmbeddings()