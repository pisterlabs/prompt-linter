import os
import argparse
from llama_index import (VectorStoreIndex, LLMPredictor,
                         SimpleDirectoryReader, ServiceContext,
                         StorageContext, load_index_from_storage)
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.node_parser import SentenceWindowNodeParser
import openai

def main(pdf_folder, index_folder):
    # Load API key from environment variable
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = input("Enter OpenAI API key: ")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    openai.api_key = OPENAI_API_KEY

    # Set context window and chunk size
    context_window = 4096
    chunk_size = 1024

    # Function to extract filename metadata
    filename_fn = lambda filename: {'file_name': filename}

    # Set up the LLMPredictor and ServiceContext
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=chunk_size, context_window=context_window)

    # Create the sentence window node parser with default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=5,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    ctx = ServiceContext.from_defaults(llm=llm, embed_model=OpenAIEmbedding(embed_batch_size=150), node_parser=node_parser)

    # Load the data from the PDF folder
    documents = SimpleDirectoryReader(pdf_folder, file_metadata=filename_fn).load_data()

    # Process and index the documents
    index = VectorStoreIndex.from_documents(documents, service_context=ctx) 

    # Save the indexed data to the INDEX folder
    index.storage_context.persist(persist_dir=index_folder)

    print("Documents processed and saved to", index_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and index PDF documents.')
    parser.add_argument('pdf_folder', type=str, help='Path to the PDF folder containing the documents.')
    parser.add_argument('index_folder', type=str, help='Path to the INDEX folder where the indexed data will be saved.')
    
    args = parser.parse_args()
    main(args.pdf_folder, args.index_folder)