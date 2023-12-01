from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.router import MultiRetrievalQAChain
from langchain.vectorstores.pgvector import PGVector
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer
from langchain import PromptTemplate
from torch import cuda, bfloat16
from flask import Flask, request
from flask import jsonify
import transformers
import warnings
import logging
import torch
import os
import re