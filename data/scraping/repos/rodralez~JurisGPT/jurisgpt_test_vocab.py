# check if all the words in a Spanish document are supported by the tokens used in the Vicuna LLM (Language Model),
from transformers import LlamaTokenizer
from langchain.document_loaders import DirectoryLoader
import os

# Define the path to the project folder
PROJECT_PATH = os.path.join(os.getenv('HOME'), 'JurisGPT')

# Change to the project folder using $HOME environment variable
os.chdir(PROJECT_PATH)

MODEL_LLM ="TheBloke/stable-vicuna-13B-HF"

tokenizer = LlamaTokenizer.from_pretrained(MODEL_LLM)

#%% LOAD

# load directory
loader = DirectoryLoader("./rawdata/laboral", glob="**/*.pdf")
documents = loader.load()

#%% EXTRACT TEXT
texts = '\n'.join([doc.page_content for doc in documents])

file_path = "texts.txt"

with open(file_path, 'w') as file:
    file.write(texts)

tokens = tokenizer.tokenize(texts)

unsupported_words = []
for token in tokens:
    if token in tokenizer.all_special_tokens:
        unsupported_words.append(token)

print ("unsupported_words: " , unsupported_words)

#%% WORD COUNT 
word_count = len(texts.split())
print("Number of words in all documents:", word_count)
print("Avr. number of words:", word_count/len(documents))


dumb = 0