import os
import chainlit as cl
from dotenv import load_dotenv
from langchain.llms import Cohere, OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Weaviate
import weaviate


load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
CLASS_NAME = "TextItem3"  # TextItem is our best index
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)

# Instantiate the client with the auth config
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth_config,
    additional_headers={"X-OpenAi-Api-Key": API_KEY},
)

weaviate_instance = Weaviate(
    client=client, index_name=CLASS_NAME, text_key="text", attributes=["source"]
)
# llm = Cohere(cohere_api_key=API_KEY)
llm = OpenAI()

template = """
You're a search agent, that helps to find answers to the questions based on the MLOPs Community Database.
Question: {question}
"""


@cl.langchain_factory(use_async=False)
def main():
    # Use RetrievalQAWithSourcesChain to return source of answer.
    # TODO: Display sources
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm, chain_type="stuff", retriever=weaviate_instance.as_retriever()
    )
    return chain


@cl.langchain_postprocess
async def postprocess(output: str):
    print(output)
    result = ""
    nono = ["none", "None."]
    if len(output["sources"]) > 0 and output["sources"] not in nono:
        sources = output["sources"].split(", ")
        src = ""
        if len(sources) > 1:
            src = []
            for source in sources:
                where_filter = {
                    "path": ["source"],
                    "operator": "Equal",
                    "valueText": source,
                }
                result = (
                    client.query.get(CLASS_NAME, ["text"])
                    .with_limit(2)
                    .with_additional(["certainty"])
                    .with_where(where_filter)
                    .do()
                )
                src.append(result["data"]["Get"]["TextItem3"][0]["text"])
        else:
            where_filter = {
                "path": ["source"],
                "operator": "Equal",
                "valueText": output["sources"],
            }
            result = (
                client.query.get(CLASS_NAME, ["text"])
                .with_limit(2)
                .with_additional(["certainty"])
                .with_where(where_filter)
                .do()
            )
            print(result)
            src = result["data"]["Get"]["TextItem3"][0]["text"]

        print(src)
        result = f"""
        Answer: {output["answer"]}
        Source: {src}
        """
    else:
        result = f"""Source cannot be verified! {output["answer"]}
        """
    await cl.Message(content=result).send()
