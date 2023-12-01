from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import fitz
import os
import json
import shutil
from langchain.llms import OpenAI, GPT4All
from datetime import datetime

# Common variables
directory_path = "./documents"
destination_path = "./summarise"
model = "gpt-4"
temperature = 1.0

# Get the list from the documents directory
def getfiles():
    # List all files in the directory
    file_names = os.listdir(directory_path)

    # Get each of the filenames in the directory and summarise
    for filename in file_names:
        summarise(filename)


# summarise each document
def summarise(filename):
    # API Keys for OPENAI
    LLM_KEY = os.environ.get("OPENAI_API_KEY")

    text = ""

    # Combine the directory and filename using os.path.join()
    source_file = os.path.join(directory_path, filename)

    #PYPDFLoader loads a list of PDF Document objects
    loader = PyPDFLoader(source_file)

    pages = loader.load()

    for page in pages:
        text+=page.page_content
    text = text.replace('\t', ' ')

    print(len(text))

    #splits a long document into smaller chunks that can fit into the LLM's
    #model's context window
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100
    )
    #create the documents from list of texts
    texts = text_splitter.create_documents([text])

    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary with key learnings\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with detailed context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary"
        "If the context isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)

    # Define the LLM
    # llm = OpenAI()
    llm = ChatOpenAI(model_name=model, temperature=temperature, request_timeout=120)

    refine_chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )
    refine_outputs = refine_chain({'input_documents': texts})
    print(refine_outputs['output_text'])
    # Save the data to file
    savedata(filename, refine_outputs['output_text'])
    # Move the original file
    relocatefile(filename)


# Save summary to a file
def savedata(filename, summary):
    # Get today's date
    today = datetime.now()
    formatted_date = today.strftime("%Y-%m-%d")

    data = [{
        "id": filename,
        "name": filename,
        "created": formatted_date,
        "owner": "openai",
        "summary": summary,
        "permissions": True,
        "ready": True
    }]

    # New filename with .json extension
    file_name = os.path.splitext(filename)[0] + ".json"
    destination_file = os.path.join(destination_path, file_name)

    # Open the file in write mode and save the data as JSON
    with open(destination_file, 'w') as json_file:
        json.dumps(data, json_file, indent=4)

    print(f"Data has been saved to {file_name}")


def relocatefile(filename):
    # Combine the directory and filename using os.path.join()
    source_file = os.path.join(directory_path, filename)

    # Combine the destination directory with the source file name to get the new file path
    new_file_path = os.path.join(destination_path, os.path.basename(source_file))

    # Move the file to the destination directory
    shutil.move(source_file, new_file_path)

    print(f"File has been moved to {new_file_path}")


# Action the file summarisation
getfiles()
