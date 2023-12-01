import json

from dotenv import load_dotenv, find_dotenv
from icecream import ic
from langchain.embeddings import OpenAIEmbeddings
from common.logger_setup import configure_logger

# another possibility is here: https://gist.github.com/szerintedmi/fdeaacc371f6bae6efb9f42b2cca734e

def create_embedding(text_string, log):

    log.info(f"text_string:{text_string}")
    try:

        embeddings = OpenAIEmbeddings()
        embeddings_result = embeddings.embed_query(text_string)
        return embeddings_result
    except Exception as e:
        log.error(f"Exception: {e}")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    log = configure_logger("embedding")
    text = "Hawaiian pizza"
    embedding = create_embedding(text, log)
    ic(embedding)


