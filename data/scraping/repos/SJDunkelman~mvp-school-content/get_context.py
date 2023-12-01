from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import openai
import re
from dotenv import load_dotenv

load_dotenv()

# Load embeddings DB
db = FAISS.load_local("faiss_index", OpenAIEmbeddings())


def pretty_print(text):
    # ANSI escape sequence for bold text
    bold = "\033[1m"

    # ANSI escape sequence for text color (cyan in this example)
    color = "\033[96m"

    # ANSI escape sequence to reset text formatting
    reset = "\033[0m"

    # Format the text with bold and color
    formatted_text = f"{bold}{color}{text}{reset}"

    # Print the formatted text
    print(formatted_text)


while True:
    query = input("Describe the section you're writing: ")
    queried_docs = db.similarity_search(query)
    if len(queried_docs) > 4:
        queried_docs = queried_docs[:4]
    for i, doc in enumerate(queried_docs):
        src = queried_docs[0].metadata['source']
        title = re.sub(r'.+/', '', src)
        pretty_print(f"\n\nDoc #{i + 1}: {title}\n\n")
        print(doc.page_content)
