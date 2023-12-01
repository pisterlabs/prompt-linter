# Author: Sam Shenoi
# Description: This file builds a class for interfacing with ChatGPT

from LangChainModel import Model
from langchain.llms.self_hosted import SelfHostedPipeline
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp

class AlpacaModel(Model):
    def __init__(self):
        self.embeddings = LlamaCppEmbeddings(model_path="./alpaca.cpp/ggml-alpaca-7b-q4.bin")
        self.llm = LlamaCpp(model_path="./alpaca.cpp/ggml-alpaca-7b-q4.bin",n_ctx=2048)
        self.rds = None

