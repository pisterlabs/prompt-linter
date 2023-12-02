## IMPORTS

from bs4 import BeautifulSoup              # Importing BeautifulSoup for HTML parsing
from bs4.element import Comment             # Importing Comment class for extracting comments from HTML
import urllib.request                       # Importing urllib.request for making HTTP requests
import streamlit as st                      # Importing streamlit for building interactive web apps
import os                                   # Importing os for accessing operating system functionalities
from dotenv import load_dotenv              # Importing load_dotenv for loading environment variables
from langchain.llms import OpenAI            # Importing OpenAI class from langchain.llms module
from langchain.prompts import PromptTemplate # Importing PromptTemplate class from langchain.prompts module
import json                                 # Importing json module for working with JSON data
from dotenv import dotenv_values            # Importing dotenv_values for loading environment variables from .env file
from googlesearch import search             # Importing search function from googlesearch module
import requests                            # Importing requests module for making HTTP requests


## SETUP ENVIRONMENT VARIABLES

load_dotenv()
env_vars = dotenv_values(".env") 


