from langchain.tools import BaseTool

import chromadb

from dotenv import dotenv_values
import openai


config = dotenv_values('.env')
openai.api_key = config['OPENAI_API_KEY']

# Open a connection to the Chroma database
chroma_path = config['CHROMA_PERSISTENT_PATH']
chroma_collection = config['CHROMA_COLLECTION']
chroma_client = chromadb.PersistentClient(path=chroma_path)

### Client Similarity Tool  #########
class ClientSimilarityTool(BaseTool):
    name = "Client Similarity Tool"
    description = "This tool is used to search for client information like balance, credit score, has a credit card, " \
                  "gender, surname, location, point earned and satisfaction score. " \
                  "Note this does not contains user names or emails." \
                  "Example query: what is the top 3 client in alabama ranked by credit score?"

    def _run(self, user_question):
        model_id = "text-embedding-ada-002"
        embedding = openai.Embedding.create(input=user_question, model=model_id)['data'][0]['embedding']
        collection = chroma_client.get_collection(name=chroma_collection)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=5
        )

        print(results)

        return results

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")