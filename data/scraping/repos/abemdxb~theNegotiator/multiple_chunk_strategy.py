"""
Builds an index for locally stored pdfs using LangChain and Pinecone. 

It involves a simple adaptation of Arize's own documentation here: https://github.com/Arize-ai/phoenix/blob/main/tutorials/build_arize_docs_index_langchain_pinecone.py

To run, you must first create an account with Pinecone and create an index in the UI with the
appropriate embedding dimension (1536 if you are using text-embedding-ada-002 like this script). You
also need an OpenAI API key. This implementation relies on the fact that the Arize documentation is
written and hosted with Gitbook. If your documentation does not use Gitbook, you should use a
different document loader.
"""
import os
import argparse
import logging
import sys
from functools import partial
from typing import Dict, List, Optional
import json
import numpy as np
import openai
import pandas as pd
import pinecone  # type: ignore
import tiktoken
from langchain.docstore.document import Document
#from langchain.document_loaders import GitbookLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import LatexTextSplitter
from langchain.vectorstores import Pinecone
from tiktoken import Encoding
from typing_extensions import dataclass_transform
from tqdm import tqdm


def load_pdf_docs(d_path: str) -> List[Document]:
    """
    Loads documentation from three pdf docs.
    """

    loader = PyPDFDirectoryLoader(d_path)

    print("loader = {}".format(loader))
    return loader.load()


def tiktoken_len(text: str, tokenizer: Encoding) -> int:
    """
    Returns the number of tokens in a text.
    """

    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def chunk_docs(documents: List[Document], embedding_model_name: str, chunk_type: str, chunk_size: int = 400, chunk_overlap: int = 20) -> List[Document]:
    """
    Chunks the documents by a specifief chunking strategy

    The original chunking strategy used in this function is from the following notebook and accompanying
    video:

    - https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/
      xx-langchain-chunking.ipynb
    - https://www.youtube.com/watch?v=eqOfr4AGLk8

    Since then we have added multiple chunk types that will be called using argparse, modified pinecone arguments to include specific subs
    """ 

    if chunk_type == 'RecursiveCharacterTextSplitter':
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=partial(
                tiktoken_len, tokenizer=tiktoken.encoding_for_model(embedding_model_name)
            ),
            separators=["\n\n", "\n", " ", ""],
        )
        return text_splitter.split_documents(documents)

    elif chunk_type == 'MarkdownTextSplitter':
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return markdown_splitter.split_documents(documents)

    elif chunk_type == 'NLTKTextSplitter':
        NLTK_text_splitter = NLTKTextSplitter()
        return NLTK_text_splitter.split_text(documents)

    elif chunk_type == 'SpacyTextSplitter':
        Spacy_text_splitter = SpacyTextSplitter()
        return Spacy_text_splitter.split_documents(documents)

    elif chunk_type == 'LatexTextSplitter':
        latex_splitter = LatexTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return latex_splitter.split_documents(documents)        

def build_pinecone_index(
    documents: List[Document], embeddings: Embeddings, index_name: str, name_space: str
) -> None:
    """
    Builds a Pinecone index from a list of documents.
    """

    Pinecone.from_documents(documents, embeddings, index_name=index_name, namespace=namespace)


def save_dataframe_to_parquet(dataframe: pd.DataFrame, save_path: str, i: int) -> None:
    """
    Saves a dataframe to parquet - ensures as we loop through combinations of namespaces that the parquet is added to instead of overwritten.
    """
    dataframe.to_parquet(save_path)
    # if i == 0 or not os.path.exists(save_path):
    #     # On the first iteration or if the file doesn't exist, create a new parquet file
    #     dataframe.to_parquet(save_path, engine='fastparquet')
    # else:
    #     # On subsequent iterations, append the data to the existing parquet file
    #     dataframe.to_parquet(save_path, engine='fastparquet', append=True)


class OpenAIEmbeddingsWrapper(OpenAIEmbeddings):
    """
    Wrapper around OpenAIEmbeddings that stores the query and document embeddings in memory.
    """

    query_text_to_embedding: Dict[str, List[float]] = {}
    document_text_to_embedding: Dict[str, List[float]] = {}

    def embed_query(self, text: str) -> List[float]:
        embedding = super().embed_query(text)
        self.query_text_to_embedding[text] = embedding
        return embedding

    def embed_documents(self, texts: List[str], chunk_size: Optional[int] = 0) -> List[List[float]]:
        embeddings = super().embed_documents(texts, chunk_size)
        for text, embedding in zip(texts, embeddings):
            self.document_text_to_embedding[text] = embedding
        return embeddings

    @property
    def query_embedding_dataframe(self) -> pd.DataFrame:
        return self._convert_text_to_embedding_map_to_dataframe(self.query_text_to_embedding)

    @property
    def document_embedding_dataframe(self) -> pd.DataFrame:
        return self._convert_text_to_embedding_map_to_dataframe(self.document_text_to_embedding)

    @staticmethod
    def _convert_text_to_embedding_map_to_dataframe(
        text_to_embedding: Dict[str, List[float]]
    ) -> pd.DataFrame:
        texts, embeddings = map(list, zip(*text_to_embedding.items()))
        embedding_arrays = [np.array(embedding) for embedding in embeddings]
        return pd.DataFrame.from_dict(
            {
                "text": texts,
                "text_vector": embedding_arrays,
            }
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pinecone-api-key", type=str, help="Pinecone API key")
    parser.add_argument("--pinecone-index-name", type=str, help="Pinecone index name")
    parser.add_argument("--pinecone-environment", type=str, help="Pinecone environment")
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key")
    parser.add_argument(
        "--output-parquet-path", type=str, help="Path to output parquet file for index"
    )
    parser.add_argument("--docs-path", type=str, help="Path to pdf files")
    parser.add_argument("--chunk-types", type=str, help="chunking_strategy") #json loads
    parser.add_argument("--chunk-sizes", type=str, help="chunk_sizes") #json loads
    parser.add_argument("--chunk-overlaps", type=str, help="chunking_overlap") #json loads
    args = parser.parse_args()

    pinecone_api_key = args.pinecone_api_key
    pinecone_index_name = args.pinecone_index_name
    pinecone_environment = args.pinecone_environment
    openai_api_key = args.openai_api_key
    output_parquet_path = args.output_parquet_path
    docs_path=args.docs_path
    chunk_types_text=args.chunk_types
    chunk_sizes_text=args.chunk_sizes
    chunk_overlaps_text=args.chunk_overlaps

    chunk_types = json.loads(chunk_types_text)
    chunk_sizes = json.loads(chunk_sizes_text)
    chunk_overlaps = json.loads(chunk_overlaps_text)
    

    openai.api_key = openai_api_key
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    p_index = pinecone.Index(pinecone_index_name)
    print("line 225 check")
    
    embedding_model_name = "text-embedding-ada-002"
    documents = load_pdf_docs(docs_path)
    print("line 228 check")
    
    loader = DirectoryLoader('theNegotiator/test/', glob="**/*.md")
    docs_md = loader.load()
    print("line 232")
    
    
    embeddings = OpenAIEmbeddingsWrapper(model=embedding_model_name)  # type: ignore
    print("line237")

    stat_dict= p_index.describe_index_stats() 
    print(stat_dict)
    namespace=""
    doc_iter= None
    i = 0
    old_df=None
    new_df=None
    print("line244")
    
    for ct in chunk_types:
        if ct == 'LatexTextSplitter' :
            print("Ignoring LatexTextSplitter as documents are pdfs")
            continue
        elif ct == "RecursiveCharacterTextSplitter":
            for cs in chunk_sizes:
                for co in chunk_overlaps:
                    #label namespace
                    namespace= f"{ct}_{cs}_{co}"
                    print(f"namespace {i}:{namespace}")
                    
                    #delete namespace if it already exists
                    if 'namespaces' in stat_dict and namespace in stat_dict['namespaces']:
                        print(f"delete occured for {namespace}")
                        p_index.delete(delete_all=True, namespace=namespace)                
                    else:
                        print(f"NO delete occured for {namespace}")
                    
                    #chunk the documents into a list of documents
                    doc_iter = chunk_docs(documents, embedding_model_name, ct, cs, co) 
                    print("doc_iter done")
                    print("doc_iter items:",f"item1::::: {doc_iter[0]}", f"item2:::::{doc_iter[1]}")
                    
                    
                    #Build the pinecone index with the document embeddings dict 
                    build_pinecone_index(doc_iter, embeddings, pinecone_index_name, namespace)
                    print("pinecone build done")

                    #Create df from embeddings
                    new_df = embeddings.document_embedding_dataframe
                    print(new_df.head())
                    print(new_df.describe())

                    #Create a diff_df that pulls only new rows from embeddings- not sure if build-index overwrites or not- not a opensource class
                    if i == 0:
                        old_df = pd.DataFrame(columns=new_df.columns)
                    
                    #Print out if new_df is an extension- for testing only
                    is_extension = all(new_df[col].isin(old_df[col].unique()).all() for col in old_df.columns)
                    if is_extension:
                        print("new_df is an extension of old_df.")
                    else:
                        print("new_df is not an extension of old_df.")
                    
                    first_column=new_df.columns[0]
                    diff_df = new_df.merge(old_df, on=first_column, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
                    diff_df["namespace"] = namespace
                    print(diff_df.head())
                    print(diff_df.describe())
                    
                    #Save to parquet file in colab and make old df = new df for next iteration - we will pull things together into one file in the colab and push to gdrive there as well.
                    final_path = output_parquet_path+namespace+".pq"
                    save_dataframe_to_parquet(diff_df, final_path,i)  
                    old_df = new_df

                    i+=1
        # elif ct == "MarkdownTextSplitter":
        #     for cs in chunk_sizes:
        #         for co in chunk_overlaps:
        #             namespace= f"{ct}_{cs}_{co}"
        #             print(f"namespace {i}:{namespace}")
        #             if 'namespaces' in stat_dict and namespace in stat_dict['namespaces']:
        #                 p_index.delete(delete_all=True, namespace=namespace)                
        #             doc_iter = chunk_docs(docs_md, embedding_model_name, ct, cs, co) 
        #             build_pinecone_index(doc_iter, embeddings, pinecone_index_name, namespace)
        #             new_df = embeddings.document_embedding_dataframe
        #             if i == 0:
        #                 old_df = pd.DataFrame(columns=new_df.columns)
        #             diff_df = new_df.merge(old_df, on=list(new_df.columns), how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
        #             diff_df["namespace"] = namespace
        #             save_dataframe_to_parquet(diff_df, output_parquet_path,i)   # can I move the choice of what to do with I here to keep things clean?
        #             old_df = new_df
        #             i+=1
        # else:
        #     namespace= f"{ct}"
        #     print(f"namespace {i}:{namespace}")
        #     if 'namespaces' in stat_dict and namespace in stat_dict['namespaces']:
        #         p_index.delete(delete_all=True, namespace=namespace)                
        #     doc_iter = chunk_docs(documents, embedding_model_name, ct) 
        #     build_pinecone_index(doc_iter, embeddings, pinecone_index_name, namespace)
        #     new_df = embeddings.document_embedding_dataframe
        #     if i == 0:
        #         old_df = pd.DataFrame(columns=new_df.columns)
        #     diff_df = new_df.merge(old_df, on=list(new_df.columns), how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
        #     diff_df["namespace"] = namespace
        #     save_dataframe_to_parquet(diff_df, output_parquet_path,i)   # can I move the choice of what to do with I here to keep things clean?
        #     old_df = new_df
        #     i+=1        
    print("Total number of namespaces:",i)

