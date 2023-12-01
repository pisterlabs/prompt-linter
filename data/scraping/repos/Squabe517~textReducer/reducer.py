import openai
import os
import tiktoken as tk
from lib.functions import readFile, split_list, write_out_response, getSummary
from dotenv import load_dotenv


# GET API KEY FROM .ENV FILE
load_dotenv()
api_key = os.environ.get("API_KEY")
# SET OPENAI API KEY
openai.api_key = api_key
# Read the file in binary mode
text_data = readFile('input.txt')


# Encode the text and see how many tokens it contains
encoding = tk.encoding_for_model('gpt-3.5-turbo')
tokenizedText = encoding.encode(text_data)

# Split the text into smaller chunks, then decode it
splitText = split_list(tokenizedText)
for i in range(len(splitText)):
  splitText[i] = encoding.decode(splitText[i])

summary = getSummary(splitText)
# Send messages to chatbot, then append responses to list

write_out_response(summary)