from dotenv import load_dotenv
load_dotenv()
import os
import bibtexparser
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
chat_LLM =  ChatOpenAI(max_tokens=1500, verbose=True, temperature=0.3)

def conv_filter(bibtex_entry):
    messages = [
    SystemMessage(content="You are a helpful assistant.")]
    # open the prompt file
    with open("prompt.txt", "r") as f:
       prompt = f.read()
    prompt = prompt + str(bibtex_entry) + "\nAnswer:"
    format_prompt = HumanMessage(
        content=prompt
    )
    messages.append(format_prompt)
    reply = chat_LLM(messages)
    reply = reply.content
    return reply

        
def convert_bib_to_bibitem(bib_file):
    with open(bib_file) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    with open('output.txt', 'w') as out_file:
        for entry in bib_database.entries:
            out_file.write(conv_filter(entry)+"\n")
    
    

# Usage
convert_bib_to_bibitem('sources.bib')
