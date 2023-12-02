import requests
from pathlib import Path
import os

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
import pprint
from rank_bm25 import BM25Okapi


def extract_behavior_descriptions_from_prompt(prompt):
    model = "gpt-3.5-turbo"
    openai.api_key = os.getenv("OPENAI_API_KEY")

    system_content = "You are a neuroscience researcher and you are interested in figuring out behavior from the methods section"

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    response = completion.choices[0].message.content

    return response


def extract_behavior_metadata_from(section):
    
    
    instructions_prompt = (
    "You are a neuroscience researcher and you are interested in figuring out behavior from the methods section of scientific papers."
    )

    prompt = (
        f"{instructions_prompt} \n"
        f"Here is the section method of a paper {section} \n" 
        "Return a list the behaviors described in the section using complete sentences and use definitions from the neurobehavior ontology. "
        "Return the list in the following format: \n"
        "First behavior: description of first behavior, second behavior : description of second behavior, ... \n"
        "Do not add any introductory sentences before the list. "
        "Do not add any notes after the list. "
        "Report only the different types of tracked behaviors. Omit any other details about the experiment. "
        "Do not add line jumps \n between the items in the list."
        "If only one behavior is described, return a list with one item."
        "If no behaviors are described, return an empty list."
    )

    response = extract_behavior_descriptions_from_prompt(prompt)

    if response != "[]":
        behaviors = response.split("\n")
        behaviors = [b for b in behaviors if b]
    else:
        behaviors = []

    return behaviors


def ground_metadata_in_ontologies(term_list: list) -> list:
    collection_name = "neuro_behavior_ontology"
    embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    qdrant_url = "https://18ef891e-d231-4fdd-8f6d-8e2d91337c24.us-east4-0.gcp.cloud.qdrant.io"
    api_key = os.environ["QDRANT_API_KEY"]
    client = QdrantClient(
        url=qdrant_url,
        api_key=api_key,
    )

    id_to_url = lambda x: f"http://purl.obolibrary.org/obo/{x.replace(':', '_')}"

    top = 10  # The number of similar vectors you want to retrieve from the database
    score_threshold = 0.5  # The minimum cosine similarity score you want to retrieve from the database
    term_embedding_list = embedding_model.embed_documents(term_list)
    term_to_embeddings = {
        term_list[i]: term_embedding_list[i] for i in range(len(term_list))
    }

    queries_response_list = []
    for term, embedding in term_to_embeddings.items():
        query_vector = embedding
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top,
            with_payload=True,
            score_threshold=score_threshold,
            with_vectors=False,
        )
        
        if search_results is None:
            continue
                
        ids = [result.payload[f"{collection_name}_id"] for result in search_results]
        names = [result.payload["name"] for result in search_results]
        text_embedded = [
            result.payload["text_for_embedding"] for result in search_results
        ]
        scores = [result.score for result in search_results]
        urls = [id_to_url(id) for id in ids]

        queries_response_list.append(
            dict(
                ids=ids,
                names=names,
                scores=scores,
                urls=urls,
                text_embedded=text_embedded,
                context=term,
            )
        )
    
    # Sort the queries_response_list by score (the scores within each query are already sorted)
    queries_response_list = sorted(
        queries_response_list,
        key=lambda item: item["scores"][0],
    )

    return queries_response_list




def rerank_with_open_ai(queries_response_list, verbose=False):
    
    ontology_terms = []

    for index in range(len(queries_response_list)):
        context = queries_response_list[index]["context"]
        names = queries_response_list[index]["names"]
        ids = queries_response_list[index]["ids"]
        urls = queries_response_list[index]["urls"]

        prompt = f"""
        We have the following behavior: {context}.

        Select the name that provides a better match for the behavior described above: {names}

        Return only the index of the selected name and nothing else. If you think that none of the names describe the behavior return -1.
        Do not return an explanation, just a number.

        """
        response = extract_behavior_descriptions_from_prompt(prompt)
        
        if verbose:
            pprint.pprint(prompt)
        
        number = int(response)
        if number != -1:
            if verbose:
                print(f"Behavior FOUND for context with {index} ")
            behavior_metadata = dict(name=names[number], id=ids[number], context=context, url=urls[number])
            ontology_terms.append(behavior_metadata)
        else:
            if verbose:    
                print(f"Behavior NOT found for context with {index} ")
        
    
    return ontology_terms
    

def rerank_with_open_ai_use_name_matching(queries_response_list, top=1, verbose=False):
    
    ontology_terms = []

    for index in range(len(queries_response_list)):
        context = queries_response_list[index]["context"]
        names = queries_response_list[index]["names"]
        ids = queries_response_list[index]["ids"]
        urls = queries_response_list[index]["urls"]
        
        prompt = f"""
        We have the following behavior: {context}.

        
        From the following list select names that provides a good  match for the behavior described above: {names} \n 

        Return a list with the names from the list above ordered by relevance from most relevant to less relevant in the following format:
        
        [most_relevant_name, second_most_relevant_name, ...]
        
        Return only the names and nothing else.
        If you think that none of the names describe the behavior return an empty list [].

        """
        response = extract_behavior_descriptions_from_prompt(prompt)
        if verbose:
            print(response)
        if response != "[]":
            names_in_query_response = response.strip("[").strip("]").replace("'", "").split(",")
            names_in_query_response = [n for n in names_in_query_response if n in names]
            if verbose:
                print(names_in_query_response)
            if names_in_query_response:
                names_index_list = [names.index(n) for n in names_in_query_response][:top]
                if verbose:
                    print(names_index_list)
                names_found = [names[n] for n in names_index_list]
                ids = [ids[n] for n in names_index_list]
                urls = [urls[n] for n in names_index_list]
                behavior_metadata = dict(names=names_found, id=ids, context=context, url=urls)
                ontology_terms.append(behavior_metadata)
                
    return ontology_terms


def rerank_with_bm25(queries_response_list, top=1):
    
    def rerank_single_entity_dictionary(entity_dictionary):

        # Tokenize the context and text_embedded
        corpus = [entity_dictionary['context']] + entity_dictionary['text_embedded']
        tokenized_corpus = [doc.split(" ") for doc in corpus]

        # Initialize BM25
        bm25 = BM25Okapi(tokenized_corpus)

        # Compute how the context is similar to each text_embedded
        scores = bm25.get_scores(tokenized_corpus[0])[1:]  # Remove the self-comparison score
        result_list = list(zip(entity_dictionary['names'], entity_dictionary['ids'], scores, entity_dictionary['urls']))
        result_list.sort(key=lambda x: x[2], reverse=True)
        
        return result_list
        
    ontology_terms = [rerank_single_entity_dictionary(entity_dict)[:top] for entity_dict in queries_response_list]
    
    return ontology_terms
