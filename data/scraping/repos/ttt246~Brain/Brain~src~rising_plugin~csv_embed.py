import os

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import json

from Brain.src.model.req_model import ReqModel
from ..common.utils import OPENAI_API_KEY


def csv_embed():
    file_path = os.path.dirname(os.path.abspath(__file__))
    loader = CSVLoader(
        file_path=f"{file_path}/guardrails-config/actions/phone.csv", encoding="utf8"
    )
    data = loader.load()

    result = list()
    for t in data:
        query_result = get_embed(t.page_content)
        result.append(query_result)
    with open(f"{file_path}/guardrails-config/actions/phone.json", "w") as outfile:
        json.dump(result, outfile, indent=2)


"""getting embed"""


def get_embed(data: str, setting: ReqModel) -> list[float]:
    embeddings = OpenAIEmbeddings(openai_api_key=setting.openai_key)
    return embeddings.embed_query(data)


if __name__ == "__main__":
    csv_embed()
