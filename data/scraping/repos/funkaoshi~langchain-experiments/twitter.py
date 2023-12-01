import argparse

from langchain.document_loaders import JSONLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore
from loguru import logger


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["favorites"] = record.get("favorite_count")
    metadata["retweets"] = record.get("retweet_count")
    metadata["created_at"] = record.get("created_at")

    return metadata


def load_tweets_into_db(path: str) -> VectorStore:
    # Load all my tweets from a twitter export into langchain documents
    loader = JSONLoader(
        file_path=path,
        jq_schema=".[].tweet",
        content_key="full_text",
        metadata_func=metadata_func,
    )
    data = loader.load()

    logger.debug(f"{len(data)} LangChain documents created from tweets.")

    # break up the text of the tweets into smaller chunks stored in a vector database
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    logger.debug(f"LangChain documents split into {len(splits)} chunks.")

    vectorstore = FAISS.from_documents(documents=splits, embedding=OllamaEmbeddings())

    # persist to disk
    vectorstore.save_local("faiss_index")

    logger.debug("VectorStore populated and ready to be queried.")

    return vectorstore


def load_tweets_from_db(path: str | None = None) -> VectorStore:
    vectorstore = None

    if path:
        # rebuild database with the supplied tweets
        vectorstore = load_tweets_into_db(path)

    if not vectorstore:
        # use existing database to load tweets
        vectorstore = FAISS.load_local("faiss_index", OllamaEmbeddings())

    logger.debug("VectorStore loaded from disk.")

    return vectorstore


def main(query: str, path: str | None):
    vectorstore = load_tweets_from_db(path)

    logger.debug("Connect to llama2 model running in Ollama")

    tweets = vectorstore.as_retriever().get_relevant_documents(query)

    tweets = [f" - {d.page_content}" for d in tweets]

    logger.debug(tweets)

    prompt_template = PromptTemplate.from_template(
        """
        These are some tweets I have written:
        
        {tweets}

        Can you use that additional context to answer the following question:

        {query}

        You should be as to the point and specific as possible, and ignore
        tweets that seem irrelevant.
        """
    )

    ollama = Ollama(base_url="http://localhost:11434", model="llama2")

    print(ollama(prompt_template.format(tweets=tweets, query=query)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline twitter query.")
    parser.add_argument("query")
    parser.add_argument("-l", "--load-tweets", action="store")
    args = parser.parse_args()

    main(args.query, args.load_tweets)
