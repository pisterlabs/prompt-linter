import PySimpleGUI as sg
import sys
import os
import argparse
import openai
import re
from pathlib import Path
import tiktoken
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, download_loader, StorageContext, load_index_from_storage, KeywordTableIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.llms import OpenAI
# for use by pyinstaller only, remove this otherwise
from llama_index.readers.llamahub_modules.file.pymu_pdf import base

# You must obtain an API key from OpenAI for use of this script:
# https://platform.openai.com/account/api-keys
#
# TODO replace this with your API key!
DEFAULT_OPENAI_API_KEY = 'YOUR_OPENAI_KEY_HERE'
enc = tiktoken.get_encoding("gpt2")

# attempt to read api key from file 'api.key' if it has not been provided in the line above
if DEFAULT_OPENAI_API_KEY == 'YOUR_OPENAI_KEY_HERE':
    try:
        with open('api.key', 'r') as key_file:
            DEFAULT_OPENAI_API_KEY = key_file.read().strip()
            if not DEFAULT_OPENAI_API_KEY:
                sg.Popup('Error', 'API key file is empty.')
                sys.exit(1)
    except FileNotFoundError:
        sg.Popup('Error', 'API key file not found.')
        # exit proram if the api key could not be found in either location
        sys.exit(1)

def sanitize_filename(filename):
    # Remove any non-alphanumeric characters (except for underscores and hyphens)
    return re.sub(r'[^\w\-_]', '', filename)

def run_model(input_file, query):
    api_key = DEFAULT_OPENAI_API_KEY
    if api_key == 'YOUR_OPENAI_KEY_HERE':
        sg.Popup('Error', "You must replace 'YOUR_OPENAI_KEY_HERE' with your actual OpenAI API key.")
        return

    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    # define LLM
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    service_context = ServiceContext.from_defaults(llm=llm)

    try:
        # PyMuPDFReader = download_loader("PyMuPDFReader")
        # for use with pyinstaller only, replace with above otherwise
        PyMuPDFReader = base.PyMuPDFReader
        loader = PyMuPDFReader()
        documents = loader.load(file_path=Path(input_file), metadata=True)

        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

        query_engine = index.as_query_engine()

        return query_engine.query(query)
    except openai.error.AuthenticationError:
        sg.Popup('Error', "An error occurred while trying to authenticate with the OpenAI API. Please ensure you've provided a valid API key.")
        return

if hasattr(sys, '_MEIPASS'):
    # PyInstaller >= 1.6
    os.chdir(sys._MEIPASS)
    os.environ["PATH"] += os.path.sep + sys._MEIPASS
elif '_MEIPASS2' in os.environ:
    # PyInstaller < 1.6 (tested on 1.5 only)
    os.chdir(os.environ['_MEIPASS2'])
    os.environ["PATH"] += os.path.sep + os.environ['_MEIPASS2']
else:
    pass

font=(sg.DEFAULT_FONT, 16)

layout = [
    [sg.Text('Input PDF file:', font=font),  sg.InputText(font=font), sg.FilesBrowse(font=font)],
    [sg.Text('Prompt:', font=font), sg.Multiline('What is the title of this document?', font=font, size=(50, 5))],
    [sg.Button('Submit', font=font)],
    [sg.Text('Answer:', font=font), sg.Multiline('output will appear here', font=font, size=(50, 10), key='output', autoscroll=True)]
]

window = sg.Window('Ask Your Document', layout, resizable=True, grab_anywhere=True)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == 'Submit':
        input_file = values[0]
        ext = input_file.split('.')[-1].lower()
        if ext != 'pdf':
            sg.Popup('Error', 'File type is not PDF')
            continue

        query = values[1]
        output = run_model(input_file, query)

        if output:
            window['output'].update(output)

window.close()
