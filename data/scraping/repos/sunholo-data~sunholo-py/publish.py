import logging
import os

from ..pubsub import PubSubManager
from ..utils.parsers import contains_url, extract_urls

from langchain.schema import Document

def publish_if_urls(the_content, vector_name):
    """
    Extracts URLs and puts them in a queue for processing on PubSub
    """
    if contains_url(the_content):
        logging.info("Detected http://")

        urls = extract_urls(the_content)
            
        for url in urls:
            publish_text(url, vector_name)


def publish_chunks(chunks: list[Document], vector_name: str):
    logging.info("Publishing chunks to embed_chunk")
    
    pubsub_manager = PubSubManager(vector_name, pubsub_topic=f"chunk-to-pubsub-embed")
        
    for chunk in chunks:
        # Convert chunk to string, as Pub/Sub messages must be strings or bytes
        chunk_str = chunk.json()
        if len(chunk_str) < 10:
            logging.warning(f"Not publishing {chunk_str} as too small < 10 chars")
            continue
        logging.info(f"Publishing chunk: {chunk_str}")
        pubsub_manager.publish_message(chunk_str)
    

def publish_text(text:str, vector_name: str):
    logging.info(f"Publishing text: {text} to app-to-pubsub-chunk")
    pubsub_manager = PubSubManager(vector_name, pubsub_topic=f"app-to-pubsub-chunk")
    
    pubsub_manager.publish_message(text)

def process_docs_chunks_vector_name(chunks, vector_name, metadata):

    pubsub_manager = PubSubManager(vector_name, pubsub_topic=f"pubsub_state_messages")
    if chunks is None:
        logging.info("No chunks found")
        pubsub_manager.publish_message(f"No chunks for: {metadata} to {vector_name} embedding")
        return None
        
    publish_chunks(chunks, vector_name=vector_name)

    msg = f"data_to_embed_pubsub published chunks with metadata: {metadata}"

    logging.info(msg)
    
    pubsub_manager.publish_message(f"Sent doc chunks with metadata: {metadata} to {vector_name} embedding")

    return metadata   