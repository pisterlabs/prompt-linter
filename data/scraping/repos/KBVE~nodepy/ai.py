#?      [IMPORT]
from llama_index import Document
import langchain
import openai
import anthropic


import os
import json
import re
import sys

#?      [DATA]
data = sys.argv[1]
jsonData = json.loads(data)
print(f"{jsonData}")
