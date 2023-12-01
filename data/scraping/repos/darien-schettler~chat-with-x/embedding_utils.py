import os
from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding, AlephAlphaSymmetricSemanticEmbedding
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, TensorflowHubEmbeddings
from langchain.embeddings import CohereEmbeddings, JinaEmbeddings, LlamaCppEmbeddings

"""
The Embedding class is a class designed for interfacing with embeddings. 
There are lots of Embedding providers (OpenAI, Cohere, Hugging Face, etc) 
  - This class is designed to provide a standard interface for all of them.

Embeddings create a vector representation of a piece of text. 
  - This is useful because it means we can think about text in the vector space
  - We use them to do things like semantic search where we look for pieces of text that are most similar in vector space

The base Embedding class in LangChain exposes two methods: `embed_documents` and `embed_query`. 
  - The largest difference is that these two methods have different interfaces: 
    - one works over multiple documents, 
    - while the other works over a single document. 
  - Besides this, another reason for having these as two separate methods is that 
    some embedding providers have different embedding methods for documents (to be searched over) vs queries (the 
    search query itself --> asymmetric embeddings).

The following integrations exist for text embeddings.
  - Aleph Alpha
    - Aleph Alpha There are two possible ways to use Aleph Alpha’s semantic embeddings. If you have texts with a
      dissimilar structure (e.g. a Document and a Query) you would want to use asymmetric embeddings. Conversely,
      for texts with comparable structures, symmetric embeddings are the suggested approach.
  - AzureOpenAI
  - Cohere
  - Fake Embeddings
  - Hugging Face Hub
  - InstructEmbeddings
  - Jina
  - Llama-cpp
  - OpenAI
  - SageMaker Endpoint Embeddings
  - Self Hosted Embeddings
  - TensorflowHub
"""


def retrieve_embedding_layer(provider, style="symmetric",
                             azure_openai_deployment_name=None, jina_model=None,
                             llama_model_path=None):
    if style == "symmetric":
        if provider == "aleph-alpha":
            return AlephAlphaSymmetricSemanticEmbedding()
        elif provider == "azure_openai":
            if azure_openai_deployment_name is None:
                raise ValueError("You must specify an embedding deployment name for AzureOpenAI embeddings")
            elif any(os.environ[x] == "" for x in ["OPENAI_API_TYPE", "OPENAI_API_BASE", "OPENAI_API_KEY"]):
                raise ValueError("You must specify OPENAI_API_TYPE, OPENAI_API_BASE, and OPENAI_API_KEY in your "
                                 "environment")
            else:
                return OpenAIEmbeddings(model=azure_openai_deployment_name)
        elif provider == "cohere":
            if os.environ["COHERE_API_KEY"] == "":
                raise ValueError("You must specify COHERE_API_KEY in your environment")
            else:
                return CohereEmbeddings()
        elif provider == "huggingface_hub":
            return HuggingFaceEmbeddings()
        elif provider == "jina":
            if os.environ["JINA_API_KEY"] == "":
                raise ValueError("You must specify JINA_API_KEY in your environment")
            elif jina_model is None:
                raise ValueError("You must specify a Jina model")
            else:
                # https://cloud.jina.ai/user/inference/model/63dca9df5a0da83009d519cd
                return JinaEmbeddings(model_name=jina_model)
        elif provider == "llama_cpp":
            if llama_model_path is None:
                raise ValueError("You must specify a llama model path")
            else:
                return LlamaCppEmbeddings(model_path=llama_model_path)

    else:
        if provider == "aleph-alpha":
            return AlephAlphaAsymmetricSemanticEmbedding()


def get_query_embedding(query, provider, style="symmetric", **kwargs):
    # Coerce the doc into a list if it isn't already
    if getattr(query, "__iter__", False) and not isinstance(query, str):
        query = "".join(query)

    embedder = retrieve_embedding_layer(provider, style=style)
    query_embedding = embedder.embed_query(query)
    return query_embedding


def get_doc_embedding(doc, provider, style="symmetric", **kwargs):
    # Coerce the doc into a list if it isn't already
    if getattr(doc, "__iter__", False) and not isinstance(doc, str):
        doc = [doc]

    embedder = retrieve_embedding_layer(provider, style=style)
    doc_embedding = embedder.embed_documents(doc)
    return doc_embedding
