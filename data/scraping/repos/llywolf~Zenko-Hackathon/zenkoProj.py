import openai
import json
import csv
import key
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import CSVLoader
from pathlib import Path
from pprint import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from googletrans import Translator
from langdetect import detect

openai.api_key = key.key
os.environ['OPENAI_API_KEY'] = key.key

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    separators= ['{', '}', '[', ']', '?', '!', '.', ',,',',', '"', ': '],
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    add_start_index = True,
)

# file_path1='/home/bogdan/hackZenko/FAQ_FDV_Zenko_V1.2.json'
# data1 = json.loads(Path(file_path1).read_text())
# file_path2='/home/bogdan/hackZenko/20230921_FetedesVendanges.json'
# data2 = json.loads(Path(file_path2).read_text())
# file_path3='/home/bogdan/hackZenko/aff_A3_2023_sd(2).json'
# data3 = json.loads(Path(file_path3).read_text())
file_path4='/home/bogdan/hackZenko/export_stands20230922 (1).csv'
#data4 = csv.reader(Path(file_path4).read_text())
with open('/home/bogdan/hackZenko/FAQ_FDV_Zenko_V1.2.json', 'r') as d1:
    data1 = d1.readlines()
with open('/home/bogdan/hackZenko/20230921_FetedesVendanges.json', 'r') as d2:
    data2 = d2.readlines()
with open('/home/bogdan/hackZenko/aff_A3_2023_sd(2).json', 'r') as d3:
    data3 = d3.readlines()
with open(Path(file_path4), 'r',encoding = "ISO-8859-1") as d4:
    data4 = d4.readlines()
with open('/home/bogdan/hackZenko/DECHETS.json', 'r') as d5:
    data5 = d5.readlines()

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
docs1 = text_splitter.create_documents(data1)
docs2 = text_splitter.create_documents(data2)
docs3 = text_splitter.create_documents(data3)
docs4 = text_splitter.create_documents(data4)
docs5 = text_splitter.create_documents(data5)
docs = docs1 + docs2 + docs3 + docs4 + docs5
# print(len(docs1))

# for item in docs:
#     if ['{', '}', '[', ']', ': '] in item:
#         docs.remove(item)
messages = docs

# print(docs)

# import
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



# sections_to_process = ["General", "Golden Gerle", "Miss and mister fdv", "Partnership", "Stand part", "Bracelet part", "Goblet Part", "Corso Part", "Transport", "Police Incident", "Firefighter incident", "Ambulance Incident", "Poisoning Incident", "Fdv Incident", "Lost Child Incident", "Home to sleep"]
# allSections = []
# for section in sections_to_process:
#     section_data = data[section]
#     for items in section_data.items():
#         for item in items:
#             allSections = allSections + section_data
#messages = [{'question': item['question'], 'answer': item['answer']} for item in vectordb]
# Define the OpenAI API key here or use environment variables

def get_api_response(messages: list) -> str:
    text = None
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=['Human:', 'AI:']
        )
        
        if response['choices'][0]['message']['role'] == 'assistant':
            text = response['choices'][0]['message']['content']
    except Exception as e:
        print('ERROR: ', e)
    
    return text


def update_list(message: str, pl: list[str]):
    pl.append(message)

def create_prompt(message: str, pl: list[dict]) -> list[dict]:
    user_message = {'role': 'user', 'content': message}
    pl.append(user_message)
    return pl

def get_bot_response(message: str, pl: list[dict]) -> str:
    prompt = create_prompt(message, pl)
    bot_response = get_api_response(prompt)
  
    if bot_response:
        assistant_message = {'role': 'assistant', 'content': bot_response}
        pl.append(assistant_message)
        bot_response = bot_response.lstrip()
    else:
        bot_response = "something went wrong"
    
    return bot_response

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)
def main():
    prompt_list = [{'role': 'system', 'content': 'You are an expert capable of meeting all the needs of a festival, both internally and externally. Central point of interaction with festival-goers. - Answer questions about tickets, schedules, and more. Provide multilingual support.'}]
    translator = Translator()
    while True:
        user_input: str = input('You: ')
        response: str = get_bot_response(user_input, prompt_list)
        # print(f'Bot: {response}')
#        print(prompt_list)
        lang = detect(user_input)
        translated = translator.translate(user_input, dest='fr')
        docs = db.similarity_search(user_input)
        translated = docs[0].page_content
        translated = translator.translate(translated, dest=lang)
        print(f'Bot: {translated}')

if __name__ == '__main__':
    main()
