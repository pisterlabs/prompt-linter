from langchain.document_loaders import PyPDFLoader
import os
import requests
from pprint import pprint

def get_brunson_pdfs():
  return [
    "players/doyle_brunson/brunson_pdfs/super_system.pdf",
    "players/doyle_brunson/brunson_pdfs/super_system_2.pdf",
  ]


