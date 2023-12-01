from dotenv import load_dotenv
import pinecone
import os
import openai

load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_API_ENV"]
)
openai.api_key = os.environ["OPENAI_API_KEY"]

index = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])


def get_metadata(text):
    # turn the query into a vector using openai
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")

    embeddings = response["data"][0]["embedding"]

    # search the index for the closest vector
    matches = index.query(
        vector=embeddings, top_k=3, include_values=False, include_metadata=True
    )

    print(matches)


def delete_by_hash(hash):
    # delete all docs with the given hash
    delete_response = index.delete(filter={"doc_hash": hash})
    print(delete_response)


# get_metadata("s514 product market fit class description syllabus")
# delete_by_hash("ae17e8b644b9fb3da4e99fbd289e2d4b8e7d183a99fdaca91443c9b1bb473023")
