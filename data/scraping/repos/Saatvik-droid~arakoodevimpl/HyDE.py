import os
import time

import PyPDF2
import gensim
import numpy as np
import openai
import redis
from dotenv import load_dotenv
from redis.commands.search.field import TagField, VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from prompt import PromptGen

load_dotenv()
prompt_gen = PromptGen()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

r = redis.Redis(
    host='redis-18848.c212.ap-south-1-1.ec2.cloud.redislabs.com',
    port=18848,
    password=os.getenv("REDIS_PASSWORD"))

INDEX_NAME = "hyde"  # Vector Index Name
DOC_PREFIX = "doc:"  # RediSearch Key Prefix for the Index

last_id = -1


def query_openai_chat_completion(messages, functions=None, function_call="auto"):
    if functions is None:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages, temperature=0.7)
    else:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages, temperature=0.7,
                                                  functions=functions, function_call=function_call)
    reply = completion.choices[0].message
    return reply


class Embedding:
    def __init__(self, filename, query, generate_examples=False):
        create_index(300)
        self.filename = filename
        self.query = query
        self.generate_examples = generate_examples
        self.examples = []

    def _read(self):
        text = ""
        name, ext = os.path.splitext(self.filename)
        if ext == ".txt":
            with open(self.filename) as f:
                text = f.read()
        elif ext == ".pdf":
            reader = PyPDF2.PdfReader(self.filename)
            num = reader.numPages
            for i in range(num):
                text += reader.getPage(i).extractText()
        else:
            return ""
        return text, self.filename

    @staticmethod
    def embed_into_db(page_content, embeddings, tag):
        pipe = r.pipeline()
        for i, (content, embedding) in enumerate(zip(page_content, embeddings)):
            # HSET
            pipe.hset(f"doc:{i}", mapping={
                "vector": embedding,
                "content": content[0],
                "doc_name": content[1],
                "tag": tag
            })
        pipe.execute()

    @staticmethod
    def get_embedding(text):
        model = gensim.models.doc2vec.Doc2Vec.load('model/new.model')
        return model.infer_vector(text.split())

    @staticmethod
    def convert_embedding_to_structure(embedding):
        return np.array(embedding).astype(np.float32).reshape(1, -1).tobytes()

    @staticmethod
    def format_query(tag, k):
        tag_query = f"(@tag:{{ {tag} }})=>"
        knn_query = f"[KNN {k} @vector $vec AS score]"
        query = Query(tag_query + knn_query) \
            .sort_by('score', asc=False) \
            .return_fields('id', 'score', 'content', 'doc_name') \
            .dialect(2)
        return query


class Character(Embedding):
    def __init__(self, filename, char_count=1000, overlap=0, generate_examples=False):
        super().__init__(filename, self.knn_doc_query_db, generate_examples)
        self.char_count = char_count
        self.overlap = overlap
        self.chunk_and_embed()

    def chunk(self, text, filename):
        page_content = [[text[i: i + self.char_count], filename] for i in
                        range(0, 1000, self.char_count - self.overlap)]
        return page_content

    def chunk_and_embed(self):
        text, filename = self._read()
        page_content = self.chunk(text, filename)
        # if self.generate_examples:
        #     self.examples = self.generate_qa_examples(page_content)
        embeddings = []
        for content in page_content:
            summary = self.summarize(content[0])
            embedding = self.get_embedding(summary.strip())
            # embedding = self.get_embedding(content[0].strip())
            embeddings.append(self.convert_embedding_to_structure(embedding))
        self.embed_into_db(page_content, embeddings, "hyde")

    @staticmethod
    def summarize(chunk):
        messages = [
            {
                "role": "user",
                "content": f"""
Summarize the following text to replace the original text with all important information left as it is.
Do not replace abbreviations with it's full forms.
{chunk}
Summary:
                """
            }
        ]
        res = query_openai_chat_completion(messages).content
        print(res)
        return res

    def knn_doc_query_db(self, query_vec, k=10):
        query = self.format_query("hyde", k)
        embedding = query_vec
        query_params = {"vec": embedding}
        ret = r.ft(INDEX_NAME).search(query, query_params).docs
        return ret


def create_index(vector_dimensions: int):
    try:
        r.ft(INDEX_NAME).dropindex(delete_documents=True)
    except:
        pass

    # schema
    schema = (
        TagField("tag"),
        VectorField("vector",  # Vector Field Name
                    "FLAT", {  # Vector Index Type: FLAT or HNSW
                        "TYPE": "FLOAT32",  # FLOAT32 or FLOAT64
                        "DIM": vector_dimensions,  # Number of Vector Dimensions
                        "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                    }
                    ),
        TextField("doc_name")
    )

    # index Definition
    definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)

    # create Index
    r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)


class Agent:
    def __init__(self, query, filename):
        create_index(1536)
        self.query = prompt_gen.SUMMARY.format(query)
        self.filename = filename

    def run(self):
        print(self.query)
        result = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.query,
            temperature=0.7,
            n=8,
            max_tokens=512,
            top_p=1,
            stop=['\n\n\n'],
            logprobs=1
        )

        # logarithmic probability sorting
        to_return = []
        for _, val in enumerate(result['choices']):
            text = val['text']
            logprob = sum(val['logprobs']['token_logprobs'])
            to_return.append((text, logprob))
        texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]

        if self.filename:
            c = Character(self.filename, 800, 100)

        print(">Generated responses:")
        text_embeddings = []
        for idx, doc in enumerate(texts):
            print(doc.strip())
            embedding = c.get_embedding(doc.strip())
            text_embeddings.append(embedding)

        # merging response embeddings
        text_embeddings = np.array(text_embeddings)
        mean_embedding = np.mean(text_embeddings, axis=0)
        mean_embedding = mean_embedding.reshape((1, len(mean_embedding)))
        result_vector = mean_embedding.tobytes()

        query_response = c.knn_doc_query_db(result_vector, 5)
        print(">Retrieved documents:")
        for res in query_response:
            print(res.id, res.doc_name, res.score)
            print(res.content)

#         ans_prompt = f"""
# Please provide an answer based solely on the provided sources.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Below are several numbered sources of information:
# ----------------
# {query_response}
#         """
        ans_prompt = f"""
You are an AI assistant whose name is DoMIno.
    - Its responses must not be vague, accusatory, rude, controversial, off-topic, or defensive.
    - It should avoid giving subjective opinions but rely on objective facts or phrases like \"in this context a human might say...\", \"some people might think...\", etc.
    - It can provide additional relevant details to answer in-depth and comprehensively covering mutiple aspects.
    - It must provide an answer based solely on the provided sources below and not prior knowledge.
    - If the documents do not provide any context refuse to answer do not create an answer for the query without documents.
    - If the full form of any abbreviation is unknown leave it as an abbreviation. Do not try to guess or infer the full form of the abrreviation. But do answer the query ussing the abbreviation without expanding it.
    - If it  doesn't know the answer, it must just say that it doesn't know and never try to make up an answer.
Below are several numbered sources of information:
----------------
{query_response}
        """
        messages = [
            {
                "role": "system",
                "content": ans_prompt
            },
            {
                "role": "user",
                "content": f"Question: {self.query}\nHelpful Answer:"
            }
        ]
        print("Final answer:\n")
        res = query_openai_chat_completion(messages).content
        print(res)


if __name__ == "__main__":
    a = Agent(
        query="What is NBFC?",
        filename="data/GUIDELINES.pdf"
    )
    a.run()
