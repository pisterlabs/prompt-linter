import os
import openai
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE") 
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

