""" 
This script is used to get the NOC code from a job description and search model, which is semantic, lexical, or mix.
"""


import os

import cohere
import pinecone
from dotenv import load_dotenv
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Pinecone
from whoosh.fields import *
from whoosh.index import open_dir
from whoosh.qparser import OrGroup, QueryParser

from config import BASEDIR

from ..content_noc21 import DETAILS_NOC21
from .translator import translate

# Load environment variables from .env file
load_dotenv()

# Get the value of an environment variable
cohere_api = os.getenv('COHERE_API_KEY')
pinecone_api = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')



# assemble semantic search results, lexical search results, and rerank results
def get_results(semantic_nocs, lexical_nocs, rerank_nocs):
    interpreted_rerank_nocs = []

    for i, hit in enumerate(rerank_nocs):
        score = hit.relevance_score
        sectionss = hit.document["text"].split("\n\n")
        noc_code, title, title_example, main_duties = sectionss
        noc_code = noc_code.replace("\n", "").replace(" ", "").split(":")[1]
        title = title.replace("\n", "")
        noc = {"noc_code": noc_code, "similarity": f"{score:.1%}"}
        interpreted_rerank_nocs.append(noc)

    results = {
        "semantic_nocs": semantic_nocs,
        "lexical_nocs": lexical_nocs,
        "rerank_nocs": interpreted_rerank_nocs,
    }

    return results


def semantic_search(query, docsearch):
    docs = docsearch.similarity_search_with_score(query, k=10)

    # return semantic search result as a list of noc codes and similarity scores
    semantic_results = []

    for i in range(len(docs)):
        doc, score = docs[i]
        sections = doc.page_content.split(
            "\n\n"
        )  # noc code, title, title examples, main duties
        noc_code, title, title_example, main_duties = sections
        noc_code = noc_code.replace("\n", "").split(" ")[2]
        title = title.replace("\n", "")

        result = {"noc_code": noc_code, "similarity": f"{score:.1%}"}
        semantic_results.append(result)

    return semantic_results


def lexical_search(ix, schema, query):
    # Define a query and search the index
    with ix.searcher() as searcher:
        # search from title, title_examples, and main_duties, until find a match
        for field in ["title", "title_examples", "main_duties"]:
            query_obj = QueryParser(field, schema, group=OrGroup).parse(query)
            results = searcher.search(query_obj)
            if len(results) > 0:
                break

        lexical_nocs = []
        for result in results:
            noc = {"noc_code": result["noc_code"], "similarity": ""}
            lexical_nocs.append(noc)

        return lexical_nocs


def get_combined_documents(semantic_nocs, lexical_nocs):
    combined_nocs = list(set(semantic_nocs + lexical_nocs))
    combined_documents = []
    for noc in combined_nocs:
        content = DETAILS_NOC21[noc]
        doc = (
            "Noc code:\n"
            + noc
            + "\n\n"
            + "Title: \n"
            + content["title"]
            + "\n\n"
            + "Title examples: \n"
            + "\n".join(content["title_examples"])
            + "\n\n"
            + "Main duties: \n"
            + "\n".join(content["main_duties"])
        )
        combined_documents.append(doc)
    return combined_documents


def get_noc_code(job_description, search_model="semantic"):
    """ """
    # if job_description is not English, translate to English
    query = (
        translate(job_description) if not job_description.isascii() else job_description
    )

    # initialize Pinecone client and whoosh index
    co = cohere.Client(cohere_api)  
    embeddings = CohereEmbeddings()

    pinecone.init(
        api_key=pinecone_api, environment=pinecone_env
    )
    docsearch = Pinecone.from_existing_index("noc2021v1", embeddings)

    schema = Schema(
        noc_code=TEXT(stored=True),
        title=TEXT(stored=True),
        title_examples=TEXT(stored=True),
        main_duties=TEXT(stored=True),
    )
    ix = open_dir(BASEDIR / "assess/nocs/ai/text_noc_index")

    # Semantic search
    semantic_nocs = (
        semantic_search(query, docsearch)
        if search_model == "semantic" or search_model == "mix"
        else []
    )

    # lexical search
    lexical_nocs = (
        lexical_search(ix, schema, query)
        if search_model == "lexical" or search_model == "mix"
        else []
    )

    # get combined nocs from semantic and lexical search
    semantic_noc_codes = [noc["noc_code"] for noc in semantic_nocs if noc]
    lexical_noc_codes = [noc["noc_code"] for noc in lexical_nocs if noc]
    combined_documents = get_combined_documents(semantic_noc_codes, lexical_noc_codes)

    rerank_nocs = co.rerank(
        query=query, documents=combined_documents, model="rerank-english-v2.0", top_n=5
    )

    results = get_results(semantic_nocs, lexical_nocs, rerank_nocs)
    return results
