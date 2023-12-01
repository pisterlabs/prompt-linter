from aml_keyvault import load_secrets
load_secrets(["OPENAI_API_KEY", "OPENAI_API_BASE", "COG_SEARCH_KEY", "COG_SEARCH_ENDPOINT"])

# set up openai api
import openai, os
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]

cog_search_patched = False

# set up cog search retriever
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.base import CallbackManager
from typing import List
from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.core.credentials import AzureKeyCredential
import mlflow
import json
import tempfile
import os
import asyncio
from patch import log_json_artifact
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

default_system_template = """Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}
"""

default_user_template = "{question}"

def parse_prompt_templates(text: str):
    from email.parser import BytesParser

    system_template = None
    user_template = None

    # Parse the message using the BytesParser
    msg = BytesParser().parsebytes(text.encode('utf-8'))

    # Iterate over the parts of the message
    for part in msg.walk():
        print(part.get_filename())
        # Check if this part has a filename attribute and that it matches the expected name
        if part.get_filename() == 'system_template.md':
            # This is the first part, do something with it
            system_template = part.get_payload(decode=True).decode('utf-8')
            # Do something with the data
        elif part.get_filename() == 'user_template.md':
            # This is the second part, do something with it
            user_template = part.get_payload(decode=True).decode('utf-8')
            # Do something with the data

    assert system_template is not None
    assert user_template is not None

    return(system_template, user_template)

def load_prompt_templates(meta_prompt: str):
    if meta_prompt:
        # load meta_prompt from file
        with open(args.meta_prompt, "r") as f:
            meta_prompt = f.read()

        if meta_prompt.startswith("Content-Type: multipart/mixed;"):
            # if meta_prompt is a multipart mime file, split it into system and user templates
            system_template, user_template = parse_prompt_templates(meta_prompt)
        else:
            # otherwise, assume it's a single template
            system_template = meta_prompt
            user_template = None
    else:
        system_template = None
        user_template = None

    return system_template, user_template

class CognitiveSearchRetriever(BaseRetriever):
    def __init__(self, endpoint: str, index_name: str, searchkey: str, top: int = 3, context_artifact_name: str = "cog_search_docs.json", verbose: bool = False):
        self.endpoint = endpoint
        self.index_name = index_name
        self.searchkey = searchkey
        self.top = top
        self.verbose = verbose
        self.context_artifact_name = context_artifact_name
        self.client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(searchkey))
        self.async_client = AsyncSearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(searchkey))
    
    def dict(self):
        return {
            "endpoint": self.endpoint,
            "index_name": self.index_name,
            "searchkey": "********",
            "top": self.top,
            "verbose": self.verbose,
            "context_artifact_name": self.context_artifact_name,
        }
    
    def _clean_documents(self, query: str, docs: List[Document]) -> List[Document]:
        cleaned_docs = []
        for i in docs:
            cleaned_docs.append(Document(page_content=i['content'], metadata={"sourcefile": i.get('sourcefile', '')}))
        if self.verbose:
            # print("cog_search_top:", self.top)
            # print("cog_search_query:", query)
            # print("cog_search_docs:", docs)
            mlflow.log_param("cog_search_top", self.top)
            mlflow.log_param("cog_search_query", query)
            docs_json = [{ 'page_content': doc.page_content, 'metadata': doc.metadata  } for doc in docs]

            # Write the dictionary to a temporary file
            log_json_artifact(docs_json, self.context_artifact_name)
        return cleaned_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.client.search(query, top=self.top)
        return self._clean_documents(query, docs)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        docs = []
        async with self.async_client:
            results = await self.async_client.search(query, top=self.top)
            async for i in results:
                docs.append(i)
        return self._clean_documents(query, docs)

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from patch import patch_langchain, log_function_call

async def rag(question: str, top: int = 3, chain_type: str = "stuff", system_template: str = None, user_template: str = None,  
        context_artifact_name: str = "cog_search_docs.json", verbose: bool = False,
        streaming_callback_manager: CallbackManager = None): 
    global cog_search_patched

    if verbose:
        # patch langchain to log function calls
        patch_langchain()
        if not cog_search_patched:
            cog_search_patched = True
            CognitiveSearchRetriever.get_relevant_documents = log_function_call(CognitiveSearchRetriever.get_relevant_documents)
        
    if system_template is None:
        system_template = default_system_template
    if user_template is None:
        user_template = default_user_template

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template),
    ]
    CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


    search_endpoint = os.environ["COG_SEARCH_ENDPOINT"]
    index_name = os.environ.get("COG_SEARCH_INDEX", "amldocs")
    searchkey = os.environ["COG_SEARCH_KEY"]
    retriever = CognitiveSearchRetriever(search_endpoint, index_name, searchkey, top=top, 
                                         context_artifact_name=context_artifact_name, verbose=verbose)

    llm_kwargs = dict(
        deployment_name="gpt-35-turbo",
        temperature=0,
        openai_api_version="2023-03-15-preview",
        verbose=True,
    )
    if streaming_callback_manager:
        llm_kwargs["streaming"] = True
        llm_kwargs["callback_manager"] = streaming_callback_manager
    llm = AzureChatOpenAI(**llm_kwargs)

    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type=chain_type,
                                    chain_type_kwargs=dict(prompt=CHAT_PROMPT),
                                    retriever=retriever)
    if streaming_callback_manager:
        return await qa.acall(question)
    return qa(question)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="What is a pipeline?")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--chain_type", type=str, default="stuff")
    parser.add_argument("--meta_prompt", type=str, default=None)
    parser.add_argument("--no-log", action='store_true')
    parser.add_argument("--score", type=str, default="./data/amldocs/scores/single_score.json")

    args = parser.parse_args()

    verbose = not args.no_log
    context_artifact_name = "cog_search_docs.json"

    if verbose:
        # mlflow.start_run()
        mlflow.log_param("question", args.question)
        mlflow.log_param("top", args.top)
        mlflow.log_param("chain_type", args.chain_type)
        mlflow.log_param("meta_prompt", args.meta_prompt)

    system_template, user_template = load_prompt_templates(args.meta_prompt)

    coroutine = rag(args.question, top=args.top, chain_type=args.chain_type, 
                    context_artifact_name=context_artifact_name,
                    system_template=system_template, user_template=user_template, verbose=verbose)
    result = asyncio.run(coroutine)

    # load the cog_search context back from MLFlow 
    if verbose:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_local_path = mlflow.artifacts.download_artifacts(f"runs:/{mlflow.active_run().info.run_id}/{context_artifact_name}", dst_path=temp_dir)
            with open(artifact_local_path, 'r') as f:
                result["context"] = json.load(f)

        log_json_artifact(result, "result.json")

    # save scores to --score output json file
    with open(args.score, "w") as f:
        json.dump(result, f, indent=4)

    print("Q:", result["query"])
    print("A:", result["result"])