# pip install "unstructured[md]"
# pip install unstructured
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import pickle
from dotenv import load_dotenv

load_dotenv()

def read_documentation():

    embedding_function = OpenAIEmbeddings()

    character_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=200,
    )

    db = Chroma(
        embedding_function=embedding_function,
        persist_directory="/Users/jacob/src/consulting/MyRareData/langchainbot_with_sources/11-Langchain-Bot",
    )

    new_memory_load = pickle.loads(
        open("/Users/jacob/src/consulting/MyRareData/langchainbot_with_sources/scrape/langchain_documents.pkl", "rb").read()
    )
    # print(new_memory_load)

    docs = character_text_splitter.split_documents(new_memory_load)
    counter = 0
    for doc in docs:
        db.add_documents([doc])
        # time.sleep(0.001)
        db.persist()
        print(counter)
        counter += 1


def main():
    read_documentation()


if __name__ == "__main__":
    main()
