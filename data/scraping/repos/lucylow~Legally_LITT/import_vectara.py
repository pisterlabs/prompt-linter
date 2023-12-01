import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.vectorstores import Vectara

load_dotenv()