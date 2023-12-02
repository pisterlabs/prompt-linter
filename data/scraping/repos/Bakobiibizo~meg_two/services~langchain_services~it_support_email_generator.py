import os
import subprocess
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.text_splitter import TokenTextSplitter
from dotenv import load_dotenv

load_dotenv()
