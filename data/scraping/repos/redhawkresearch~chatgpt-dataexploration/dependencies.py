import os
import sys
import psutil 

from io import BytesIO, StringIO
import requests

####
# Slackbot Imports
####
import slack_sdk as slack
from slack_sdk import WebClient

from slack_bolt import App, Ack, Respond
from slack_bolt.adapter.socket_mode import SocketModeHandler

####
# LangChain Imports
####
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
# Use pandas dataframes
import pandas as pd
from pandas import DataFrame

# Use ChatGPT conversationally with context (data)
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

