import json
import requests
from pathlib import Path
import os

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

root_path = Path(__file__).parent.parent
data_dir = root_path / "data"
assert data_dir.is_dir()
file_path = data_dir / "training_data.json"
dandiset_training_data = json.load(file_path.open())


def generate_prompt_from_dandiset_id(dandiset_id: str) -> str:
    dandiset_fields_metadata = dandiset_training_data[dandiset_id]["metadata"]
    abstract = dandiset_training_data[dandiset_id]["abstract"]

    prompt = generate_prompt_for_example(abstract, dandiset_fields_metadata)
    return prompt


def generate_prompt_for_example(abstract: str, expected_metadata: dict) -> str:
    keys_to_extract = ["species_names", "anatomy", "approach_names", "measurement_names"]
    metadata_to_extract = {k: expected_metadata[k] for k in keys_to_extract}
    prompt = f"""The abstract of the paper is:
    {abstract} 
    Information: {json.dumps(metadata_to_extract)}
    """
    return prompt


def generate_prompt_examples(dandiset_ids: list) -> str:
    examples_prompt = ""
    for i, dandiset_id in enumerate(dandiset_ids):
        examples_prompt += (
            f"\n- Example {i+1}: {generate_prompt_from_dandiset_id(dandiset_id)}"
        )
    return examples_prompt


def get_crossref_abstract(doi):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "message" in data and "abstract" in data["message"]:
            abstract_message = data["message"]["abstract"]
            start_index = abstract_message.find("<jats:p>") + len("<jats:p>")
            end_index = abstract_message.find("</jats:p>")
            abstract_text = abstract_message[start_index:end_index]

            # Extract the text between the tags
            cleaned_abstract_text = (
                abstract_text.replace("<jats:sup>", "")
                .replace("</jats:sup>", "")
                .replace("<jats:italic>", "")
                .replace("</jats:italic>", "")
            )

            return cleaned_abstract_text

        else:
            return None
    else:
        return None


def generate_task_prompt_from_abstract(abstract: str) -> str:
    prompt = f"""The abstract of the paper is:
    {abstract} 

    Fill as in the examples:
    Information: {{}}
    In the format of the previous response. If some information is missing, leave it blank.
    """
    return prompt


def query_metadata(prompt):
    system_content = "You are a neuroscience researcher and you are interested in figuring relevant metadata from abstracts"

    model = "gpt-3.5-turbo"
    openai.api_key = os.getenv("OPENAI_API_KEY")
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


def ground_metadata_in_ontologies(plain_metadata):
    
    embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    url = "https://c1490259-dfe4-4a49-8712-24f690d450f6.us-east-1-0.aws.cloud.qdrant.io:6333"
    api_key =os.environ["QDRANT_API_KEY"]
    client = QdrantClient(
    url=url,
    api_key=api_key,
    )
    
    grounded_meatdata = _ground_metadata_in_ontologies_uberon(client, embedding_model, plain_metadata)
    grounded_metadat = _ground_metadata_in_ontologies_ncbi_taxon(client, embedding_model, plain_metadata)
    
    return grounded_meatdata


def _ground_metadata_in_ontologies_uberon(client, embedding_model, plain_metadata):
    collection_name = "uberon"
    id_to_url = lambda x: f"http://purl.obolibrary.org/obo/{x.replace(':', '_')}"

    anatomy_terms_list =  plain_metadata["anatomy"]
    if anatomy_terms_list == []:
        return plain_metadata
    
    term_embedding_list = embedding_model.embed_documents(anatomy_terms_list)
    anatomy_term_to_embedings = {anatomy_terms_list[i]: term_embedding_list[i] for i in range(len(anatomy_terms_list))}

    top = 1  # The number of similar vectors you want to retrieve
    term_to_identifiers_dict = {}
    for anatomy_term, embedding in anatomy_term_to_embedings.items():
        
        query_vector = embedding
        search_results = client.search(collection_name=collection_name, query_vector=query_vector, limit=top, with_payload=True, with_vectors=False)
        
        uberon_ids = [result.payload["uberon_id"] for result in search_results]
        uberon_urls = [id_to_url(uberon_id) for uberon_id in uberon_ids]
        
        term_to_identifiers_dict[anatomy_term] = (uberon_ids[0], uberon_urls[0])
        
    
    plain_metadata[f"{collection_name}_identifiers"] = [term_to_identifiers_dict[anatomy_term][0] for anatomy_term in anatomy_terms_list]
    plain_metadata[f"{collection_name}_urls"] = [term_to_identifiers_dict[anatomy_term][1] for anatomy_term in anatomy_terms_list]
    
    return plain_metadata
    

def _ground_metadata_in_ontologies_ncbi_taxon(client, embedding_model, plain_metadata):
    collection_name = "ncbi_taxon"
    id_to_url = lambda x: f"http://purl.obolibrary.org/obo/{x.replace(':', '_')}"

    term_list =  plain_metadata["species_names"]
    if term_list == []:
        return plain_metadata
    
    term_embedding_list = embedding_model.embed_documents(term_list)
    term_to_embeddings = {term_list[i]: term_embedding_list[i] for i in range(len(term_list))}

    top = 1  # The number of similar vectors you want to retrieve
    term_to_identifiers_dict = {}
    for term, embedding in term_to_embeddings.items():
        
        query_vector = embedding
        search_results = client.search(collection_name=collection_name, query_vector=query_vector, limit=top, with_payload=True, with_vectors=False)
        
        ids = [result.payload["ncbi_taxon_id"] for result in search_results]
        urls = [id_to_url(id) for id in ids]
        
        term_to_identifiers_dict[term] = (ids[0], urls[0])
        
    
    plain_metadata[f"{collection_name}_identifiers"] = [term_to_identifiers_dict[term][0] for term in term_list]
    plain_metadata[f"{collection_name}_urls"] = [term_to_identifiers_dict[term][1] for term in term_list]
    
    return plain_metadata

def parse_response_to_dict(response):
    start_index = response.find("{")
    end_index = response.rfind("}")
    json_string = response[start_index : end_index + 1]
    response_dictionary = json.loads(json_string)

    return response_dictionary

def infer_metadata(prompt):
    response = query_metadata(prompt)

    try:
        response_dict = parse_response_to_dict(response)
        return response_dict
    except:
        return response