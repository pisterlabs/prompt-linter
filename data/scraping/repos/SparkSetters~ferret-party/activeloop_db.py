from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings
import os
os.environ['ACTIVELOOP_TOKEN']


def activeloop_db():
    my_activeloop_org_id = "giantpineapplestatue"
    my_activeloop_dataset_name = "county_policy"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    return db
