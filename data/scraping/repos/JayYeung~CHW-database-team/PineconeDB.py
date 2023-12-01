import openai
import pinecone
from time import sleep
import uuid
import os
from local_secrets import pinecone_api_key, pinecone_environment

openai.api_key = os.environ.get("OPENAI_API_KEY")


class Database:
    def __init__(self, index_name, pinecone_api_key, pinecone_environment, embed_model):
        self.embed_model = embed_model

        print('Connecting to Pinecone')
        pinecone.init(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )

        if index_name not in pinecone.list_indexes():
            sample_embedding = self.create_embeddings(["Sample document text"])
            print(f'Creating {index_name} index')
            pinecone.create_index(
                index_name,
                dimension=len(sample_embedding),
                metric='dotproduct'
            )
            print('Done')
        else:
            print(f'Connecting to existing {index_name} index')

        self.index = pinecone.Index(index_name=index_name)  # Store the index object as an attribute
        print(pinecone.describe_index(index_name))

    def create_embeddings(self, texts):
        try:
            res = openai.Embedding.create(input=texts, engine=self.embed_model)                    
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = openai.Embedding.create(input=texts, engine=self.embed_model)
                    done = True
                except:
                    pass
        return res['data'][0]['embedding']

    def insert(self, message, metadata=None, namespace=None):
        embedding = self.create_embeddings(message)
        data = {
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": message
            }
        }
        if metadata: 
            data["metadata"].update(metadata)
        if namespace: 
            self.index.upsert(vectors=[data],
                            namespace=namespace)
        else: 
            self.index.upsert(vectors=[data])
        print("metadata: ", data["metadata"])

    def retrieve(self, message):
        query_embedding = self.create_embeddings(message)
        results = self.index.query(vector = query_embedding, 
                                   include_metadata = True, top_k=10)
        return results

INDEX_NAME = "test"
PINECONE_API_KEY = pinecone_api_key
PINECONE_ENVIRONMENT = pinecone_environment
EMBED_MODEL = "text-embedding-ada-002"

db = Database(INDEX_NAME, PINECONE_API_KEY, PINECONE_ENVIRONMENT, EMBED_MODEL)

#message = "Hello, this is a test message from hi. "
#db.insert(message)

message = "Hello, this is another test message from abc."
db.insert(message)

message = "oh no do I have fever symptoms?"
db.insert(message, namespace='test')

# message = 'I LOve CATS CATS CATS'
# db.insert(message)

query_message = "i have a fever"
retrieval_results = db.retrieve(query_message)

print("Retrieval Results:")
print(retrieval_results)