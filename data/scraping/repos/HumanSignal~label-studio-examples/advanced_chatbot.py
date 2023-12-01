import os
import openai
import gradio as gr
import argparse
import logging
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Logging setup
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)


def main(persist_dir):

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1000),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True)
    
    def predict(message, history):
        history_openai_format = []
        history_openai_format.append({"role": "user", "content": message})
        response = qa_chain({"query": str(history_openai_format)})
        return response['result']

    gr.ChatInterface(predict).queue().launch(debug=True) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameterized script for chat interface.")
    parser.add_argument("--persist_dir", default="pd", help="Persist directory for vectorstore.")
    
    args = parser.parse_args()

    main(args.persist_dir)
