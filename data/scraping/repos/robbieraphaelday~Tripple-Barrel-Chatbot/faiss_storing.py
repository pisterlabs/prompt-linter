import os
import pickle
from langchain.text_splitter import CharacterTextSplitter
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


with open('openai_api.txt', 'r') as f:
    openai_apikey = f.read().strip()

os.environ['OPENAI_API_KEY'] = openai_apikey

def get_text_content(path):
    with open(path, 'r') as f:
        content = f.read()
    return content

def main():

    # Get all text files in the 'text_docs' directory
    text_files = [f for f in os.listdir('text_docs') if f.endswith('.txt')]
    print(f"\nFound {len(text_files)} text files\n")

    # Concatenate the contents of all text files
    corpus = ""
    for text_file in text_files:
        corpus += get_text_content(os.path.join('text_docs', text_file))

    print("\nFinished processing all text files\n")

    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(        
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,  # striding over the text
        length_function=len,
    )

    texts = text_splitter.split_text(corpus)
    print(f"\n\nText split into {len(texts)} chunks\n")

    # Download embeddings from OpenAI
    print("\nDownloading embeddings from OpenAI\n")
    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)
    print("\nCreated FAISS index\n")

    # Save embeddings to a pickle file
    with open('embeddings.pickle', 'wb') as f:
        pickle.dump(docsearch, f)

    print("\nEmbeddings saved to embeddings.pickle\n")

if __name__ == "__main__":
    main()
