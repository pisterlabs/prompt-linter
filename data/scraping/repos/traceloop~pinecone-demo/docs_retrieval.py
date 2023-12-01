import os
import openai
import pinecone

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

Traceloop.init(disable_batch=True)

openai.api_key = os.getenv('OPENAI_API_KEY')
embed_model = "text-embedding-ada-002"

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index_name = 'gpt-4-langchain-docs-fast'
index = pinecone.GRPCIndex(index_name)


@task(name="retrieve_docs")
def retrieve_docs(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts (including the questions)
    return index.query(xq, top_k=5, include_metadata=True)


def augment_query(query, pinecone_res):
    contexts = [item['metadata']['text'] for item in pinecone_res['matches']]
    return "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query


@task(name="query_llm")
def query_llm(augmented_query):
    primer = """You are Q&A bot. A highly intelligent system that answers
user questions based on the information provided by the user above
each question. If the information can not be found in the information
provided by the user you truthfully say "I don't know".
"""

    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query}
        ]
    )


@workflow(name="ask_question")
def ask_question(question):
    relevant_docs = retrieve_docs(question)
    augmented_query = augment_query(question, relevant_docs)
    return query_llm(augmented_query).choices[0].message.content


print(ask_question("how do I build an agent with LangChain?"))
