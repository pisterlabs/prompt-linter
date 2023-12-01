# youtube_transcripts_analysis.py

import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter

# Define the path to your transcript files
transcript_dir = 'transcripts'

# Function to read each transcript file
def read_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Process and split transcripts into chunks
def process_and_split_transcripts():
    texts = []
    for filename in os.listdir(transcript_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(transcript_dir, filename)
            transcript_content = read_transcript(file_path)
            texts.append(transcript_content)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    return [text_splitter.split_text(text) for text in texts]

# OpenAI API Key Setup

openai.api_key = os.environ["OPENAI_API_KEY"]

# Embedding Transcripts
def embed_transcripts(texts):
    embeddings = OpenAIEmbeddings()
    return Chroma.from_texts(texts, embeddings).as_retriever()

# Main function to process and query transcripts
def main():
    # Process and split transcripts
    split_texts = process_and_split_transcripts()

    # Flatten the list of lists if necessary
    flattened_texts = [item for sublist in split_texts for item in sublist]

    # Embed transcripts
    docsearch = embed_transcripts(flattened_texts)

    # Example query
    query = "Based on all of the transcripts, summarize who is Aaron LeBauer and what does he know about physical therapy?"
    docs = docsearch.get_relevant_documents(query)

    # Setting up the Chat Model for LangChain
    chat_model = ChatOpenAI(model_name="gpt-4-1106-preview")

    # Setting up the QA chain
    chain = load_qa_chain(llm=chat_model, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)

    print("Answer:", answer)

if __name__ == "__main__":
    main()
