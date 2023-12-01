#!/usr/bin/env python

## Life AI Document Injection module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import warnings
import logging
import time
import json
import logging
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from constants import CHROMA_SETTINGS
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
import re
import nltk  # Import nltk for sentence tokenization
import spacy ## python -m spacy download en_core_web_sm

load_dotenv()

warnings.simplefilter(action='ignore', category=Warning)

# Download the Punkt tokenizer models (only needed once)
nltk.download('punkt')

def extract_sensible_sentences(text):
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Process the text with spaCy
    doc = nlp(text)

    # Filter sentences based on some criteria (e.g., length, structure)
    sensible_sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 3 and is_sensible(sent.text)]

    logger.debug(f"Extracted {text} into sensible sentences: {sensible_sentences}\n")

    return sensible_sentences

def is_sensible(sentence):
    # Implement a basic check for sentence sensibility
    # This is a placeholder - you'd need a more sophisticated method for real use
    return not bool(re.search(r'\b[a-zA-Z]{20,}\b', sentence))

def clean_text(text):
    # truncate to 800 characters max
    text = text[:args.max_size]
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove image tags or Markdown image syntax
    text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<img.*?>', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove any inline code blocks
    text = re.sub(r'`.*?`', '', text)
    
    # Remove any block code segments
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove special characters and digits (optional, be cautious)
    text = re.sub(r'[^a-zA-Z0-9\s.?,!]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())

    # Extract sensible sentences
    sensible_sentences = extract_sensible_sentences(text)
    text = ' '.join(sensible_sentences)

    return text

def main():
    embeddings = HuggingFaceEmbeddings(model_name=args.embeddings)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path="db")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": args.doc_count})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model=args.model, max_tokens=args.max_tokens, backend='gptj', n_batch=8, callbacks=callbacks, verbose=False)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    while True:
        header_message = receiver.recv_json()

        # context
        history = []
        # check if history is an array, if not make it one
        if 'history' in header_message and type(header_message['history']) is not list:
            history = [header_message['history']]
        elif 'history' in header_message:
            history = header_message['history']
        
        # message
        message = ""
        if 'message' in header_message:
            message = header_message['message']
        else:
            message = ""

        message = clean_text(message)

        logger.debug(f"received message: {message} in context: {json.dumps(history)} {json.dumps(header_message)}\n")

        # look up in chroma db
        logger.info(f"looking up {message} in chroma db...\n")
        res = qa(message[:800])
        if res is None:
            logger.error(f"Error getting answer from Chroma DB: {res}")
            return None
        if 'result' not in res:
            logger.error(f"Error getting answer from Chroma DB: {res}")
            return None
        if 'source_documents' not in res:
            logger.error(f"Error getting answer from Chroma DB: {res}")
            return None
        logger.debug(f"got answer: {res['result']}.\n")
        answer, docs = res['result'], res['source_documents']
        for document in docs:
            logger.debug(f"got document: {document.metadata}\n")
            source_doc = document.metadata["source"]
            context_add = f"{clean_text(source_doc)} {clean_text(document.page_content)}"
            logger.info(f"Adding to context from source {source_doc}: {context_add}\n")
            history.append(f"{context_add}")

        logger.info(f"Sending message from answer: {answer} with history: {json.dumps(history)}\n")
        header_message['history'] = history

        sender.send_json(header_message)
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=8000, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=1500, required=False, help="Port for sending image output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending image output")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--max_size", type=int, default=2048, required=False, help="Maximum size of text to process")
    parser.add_argument("--max_tokens", type=int, default=32768, required=False, help="Maximum tokens to process")
    parser.add_argument("--doc_count", type=int, default=2, required=False, help="Number of documents to return")
    parser.add_argument("--model", type=str, default="models/ggml-all-MiniLM-L6-v2-f16.bin", required=False, help="GPT model to use")
    parser.add_argument("--embeddings", type=str, default="all-MiniLM-L6-v2", required=False, help="HuggingFace embedding model to use")

    args = parser.parse_args()

    LOGLEVEL = logging.INFO

    if args.loglevel == "info":
        LOGLEVEL = logging.INFO
    elif args.loglevel == "debug":
        LOGLEVEL = logging.DEBUG
    elif args.loglevel == "warning":
        LOGLEVEL = logging.WARNING
    else:
        LOGLEVEL = logging.INFO

    log_id = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f"logs/docInject-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('docInject')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    logger.info("connected to ZMQ in: %s:%d\n" % (args.input_host, args.input_port))
    receiver.bind(f"tcp://{args.input_host}:{args.input_port}")
    #receiver.subscribe("")

    sender = context.socket(zmq.PUSH)
    logger.info("binded to ZMQ out: %s:%d\n" % (args.output_host, args.output_port))
    sender.bind(f"tcp://{args.output_host}:{args.output_port}")

    main()

