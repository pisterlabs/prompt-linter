import warnings

warnings.filterwarnings("ignore")
import os, openai, cohere
import gradio as gr
from pathlib import Path
from langchain.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_CLUSTER_URL = os.environ["QDRANT_CLUSTER_URL"]
QDRANT_COLLECTION_NAME = os.environ["QDRANT_COLLECTION_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
prompt_file = "prompt_template.txt"


def pdf_loader(pdf_file):
    yield "Extracting contents from PDF document..."

    loader_mu = PyMuPDFLoader(pdf_file.name)
    pages = loader_mu.load()
    docs = []
    for i in range(len(pages)):
        raw_page_content = pages[i].page_content
        metadata_source = {"source": str(i + 1)}
        doc = Document(
            page_content=pages[i].page_content, metadata={"source": str(i + 1)}
        )
        docs.append(doc)

    yield "Splitting contents into chunks of text..."
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=1024,
        chunk_overlap=64,
        separators=["\n\n", "\n", " "],
    )

    docs_splitter = text_splitter.split_documents(docs)
    cohere_embeddings = CohereEmbeddings(model="large", cohere_api_key=COHERE_API_KEY)

    yield "Uploading chunks of text into Qdrant..."
    qdrant = Qdrant.from_documents(
        docs_splitter,
        cohere_embeddings,
        url=QDRANT_CLUSTER_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    with open(prompt_file, "r") as file:
        prompt_template = file.read()

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question", "context"]
    )

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY
    )
    global qa
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=qdrant.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
    )

    yield "Success! You can now click on the 'AI Assistant' tab to interact with your document"


def chat(chat_history, query):
    res = qa.run(query)
    progressive_response = ""

    for ele in "".join(res):
        progressive_response += ele + ""
        yield chat_history + [(query, progressive_response)]


with gr.Blocks() as demo:
    gr.HTML(
        """<h1>Welcome to AI PDF Assistant</h1>"""
    )
    gr.Markdown(
        "AI Assistant for PDF documents. Upload your pdf document, click 'Process PDF docs' and wait for success confirmation message.<br>"
        "After success confirmation, click on the 'AI Assistant' tab to interact with your document.<br>"
        "Type your query, and  hit enter. Click on 'Clear Chat History' to delete all previous conversations."
    )

    with gr.Tab("Upload/Process PDF documents"):
        text_input = gr.File(label="Upload PDF file", file_types=[".pdf"], type="file")
        text_output = gr.Textbox(label="Status...")
        text_button = gr.Button("Process PDF docs!")
        text_button.click(pdf_loader, text_input, text_output)

    with gr.Tab("AI Assistant"):
        chatbot = gr.Chatbot()
        query = gr.Textbox(
            label="Type your query here, then press 'enter' and scroll up for response"
        )
        clear = gr.Button("Clear Chat History!")
        query.submit(chat, [chatbot, query], chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)


demo.queue().launch()
