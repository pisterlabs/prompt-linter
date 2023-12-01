from pymongo import MongoClient
import pinecone
import os
from openai import OpenAI

# from dotenv import load_dotenv
# load_dotenv()

class MyOpenAI:
    def __init__(self):
        self.client = OpenAI()
        self.cost = {
            'chat': 0,
            'embedding': 0,
        }
        self.price = {
            'chat': 0.001/1000,
            'embedding': 0.0001/1000,
        }

    def get_embedding(self, text):
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
            )
            self.cost['embedding'] += response.usage.total_tokens
            return response.data[0].embedding
        except Exception as e:
            print(e)
            return None

    def chat(self, text, prompt):
        try:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ])
            self.cost['chat'] += completion['usage']['total_tokens']
            print(completion.choices[0].message)
        except Exception as e:
            print(e)
            return None
        
    def get_cost(self):
        chat_cost = self.cost['chat'] * self.price['chat']
        embedding_cost = self.cost['embedding'] * self.price['embedding']
        return chat_cost + embedding_cost


class MyMongo:
    _instance = None

    def __new__(cls, uri=None):
        if cls._instance is None:
            cls._instance = super(MyMongo, cls).__new__(cls)
            cls.client = MongoClient(uri)
            return cls._instance

    def __del__(self):
        self.client.close()

    def get_db(self, db_name):
        return self.client[db_name]


class PineconeM:
    def __init__(self, pinecone_key):
        pinecone.init(api_key=pinecone_key,
                      environment="gcp-starter")
        self.index = pinecone.Index("newssnap-test")

    def insert(self, id, vector, metadata):
        try:
            self.index.upsert([(id, vector, metadata)])
        except Exception as e:
            print(e)
            return False

        return True

    def query(self, vector, k_top=10, filter=None, include_metadata=True, include_values=False):
        try:
            res = self.index.query(
                vector=vector,
                top_k=k_top,
                filter=filter,
                include_metadata=include_metadata, include_values=include_values)
        except Exception as e:
            print('error', e)
            return None

        return res['matches']


mypinecone = PineconeM(os.getenv('PINECONE_KEY'))
mymongo = MyMongo(os.getenv('MONGODB_URI'))
myopenai = MyOpenAI()