import argparse
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from typing import List
from langchain.schema import Document
import curses

os.environ['OPENAI_API_KEY'] = open('openAI.token').read().strip()

class Genie:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = TextLoader(self.file_path)
        self.documents = self.loader.load()
        self.texts = self.text_split(self.documents)
        self.vectordb = self.embeddings(self.texts)
        self.genie = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=self.vectordb.as_retriever())

    @staticmethod
    def text_split(documents: TextLoader):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    @staticmethod
    def embeddings(texts: List[Document]):
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(texts, embeddings)
        return vectordb

    def ask(self, query: str):
        return self.genie.run(query)

def main(stdscr):
    curses.curs_set(0)  # Hide the cursor
    stdscr.clear()
    
    # Specify the folder where the text files are located
    folder_path = "storage" 
    
    # List the available text files in the "storage" folder
    text_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    stdscr.addstr(2, 2, "Choose a text file:")
    
    for i, text_file in enumerate(text_files):
        stdscr.addstr(i + 4, 4, f"{i + 1}. {text_file}")
    
    stdscr.addstr(len(text_files) + 5, 2, "Vælg hvilken database, du vil spørge (indtast tallet): ")
    stdscr.refresh()

    text_file_choices = {str(i + 1): text_file for i, text_file in enumerate(text_files)}

    while True:
        key = stdscr.getkey()
        if key in text_file_choices:
            text_file = text_file_choices[key]
            break

    stdscr.clear()
    stdscr.addstr(2, 2, f"Valgt database: {text_file}")
    stdscr.addstr(4, 2, "Hvad vil du spørge om?: ")
    stdscr.refresh()
    curses.echo()
    query = stdscr.getstr(5, 2, 60)
    curses.noecho()

    # Specify the full path to the selected text file
    selected_file_path = os.path.join(folder_path, text_file)
    
    genie = Genie(selected_file_path)
    answer = genie.ask(query.decode())
    stdscr.clear()
    stdscr.addstr(2, 2, "Valgt database: " + text_file)
    stdscr.addstr(4, 2, "Spørgsmål: " + query.decode())
    stdscr.addstr(6, 2, "Svar: ")
    stdscr.addstr(8, 4, answer)
    stdscr.refresh()
    stdscr.getch()

if __name__ == "__main__":
    curses.wrapper(main)
