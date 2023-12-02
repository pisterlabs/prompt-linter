import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def timed_function(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

embedding_model = "multi-qa-MiniLM-L6-cos-v1"

embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)

# Initializing VectorDB connection which was pre-populated with Knowledge Asset vector embeddings
vector_db_entities = Milvus(
    collection_name="EntityCollection",
    embedding_function=HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1"),
    connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
            "secure": True,
        },
)

vector_db_relations = Milvus(
    collection_name="RelationCollection",
    embedding_function=HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1"),
    connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
            "secure": True,
        },
)

@timed_function
def extract_entities_relations(question: str) -> (list, list):
    try:
        # Call the OpenAI ChatCompletion API
        completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
            "role": "system",
            "content": "You will receive a natural language prompt. Your task is to analyze the prompt and extract the key entities, attributes, and relations needed to build a SPARQL query. Entities are specific nodes in the graph database. Attributes of entities, such as 'industry' or 'foundedYear', should also be considered as relations. Relations are the connections between entities, formatted as (subject From Prompt(and type), predicate From Prompt, object or value form prompt (and type)). Include entity attributes from the prompt with Entities, and add them as relations. Organize your response in JSON format, listing the entities under 'Entities:' and the relations as triples under 'Relations:'. Ensure relations are identified in the format of '(subject(subjectType), predicate, object/value(objectType/valueType))'. Example: {'Entities:': {'Entity1': {'@type': 'Organization', 'Revenue': {'value': 10, 'comparator': '>'}, 'industry': 'Agriculture'}, 'Entity2': {'@type': 'Person', 'WorksFor': 'Entity1'} }, 'Relations:': ['(Entity1(Organization), Revenue, {'value': 10, 'comparator': '>'}(number)), (Entity1(Organization), industry, 'agriculture'(string)), (Entity2(Person), investedIn, Entity1(Organization))']}."
            },
            {
                "role": "user",
                "content": f"Analyze this prompt and identify the entities, attributes, and relations needed to construct a SPARQL query that answers the prompt: '{question}'"
            }  
        ]
        )

        # Extract the response content
        extracted_content = completion.choices[0].message.content
        data = json.loads(extracted_content)
        print(data)
        entities = []
        relations = []

        # Process entities
        for entity_name, attributes in data["Entities:"].items():
            entity_str = f"{entity_name}: "
            for attr, value in attributes.items():
                entity_str += f"{attr}:{value}; "
            entities.append(entity_str.rstrip('; '))  # Remove trailing semicolon

        relation_pattern = re.compile(r'\((.+?)\((.+?)\), (.+?), (.+)\((.+?)\)\)')
        # Process relations
        for relation in data["Relations:"]:
            match = relation_pattern.match(relation)
            if match:
                subject = match.group(1)
                subject_type = match.group(2)
                predicate = match.group(3)
                object_or_value = match.group(4)
                object_or_value_type = match.group(5)
                relations.append(f"({subject}({subject_type}), {predicate}, {object_or_value}({object_or_value_type}))")

        # Deduplication if necessary
        entities = list(set(entities))
        relations = list(set(relations))

        print('Entities:', entities)
        print('Relations:', relations)

        return entities, relations

    except Exception as e:
            print(f"Error occurred: {e}")
            return [], []  # Return empty lists if an error occurs



@timed_function
def EntityMatchAnswer(question):
 #match quations to entities
        print("1")
        docs = vector_db_entities.similarity_search(question)
        print("2")
        all_documents = []

        for doc in docs:
            document_dict = {
                'page_content': doc.page_content,
                'metadata:': {
                    **doc.metadata
                }
            }
            all_documents.append(document_dict)


        # If we want to, we can submit the extracted results further to an LLM (OpenAI in this case) to obtain a summary of the extracted information
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system", 
                    "content": f"You receive a question and some JSON, and you answer the question based on the information found in the JSON. You do not mention the JSON in the response, but just produce an answer"
                },
                {
                    "role": "user", 
                    "content": f"Answer the question: {question} based on the following json: {json.dumps(all_documents)}",},
            ])
        return all_documents, response


@timed_function
def construct_sparql_query_openai(question: str, matched_entities: list, matched_relations: list) -> str:
    # Extract string representations from Document objects
    formatted_entities = "; ".join([str(doc) for doc in matched_entities])
    formatted_relations = "; ".join([str(doc) for doc in matched_relations])

    prompt = f"""
    Construct a SPARQL query to answer the following question based on the given entities and relations. 
    Pay attention to the directionality of the predicate given the subject types and object types in Mateched Relations. 
    When using variables that represent literals, cast the variable accordingly, for example if ?foundedYear is type http://www.w3.org/2001/XMLSchema#integer use xsd:integer(?foundedYear) in the query.
    Do not provide any additional explanation or content, only the SPARQL query itself.

    Question: {question}
    Matched Entities: {formatted_entities}
    Matched Relations: {formatted_relations}
    """

    response = client.chat.completions.create(model="gpt-4-1106-preview",
                                              temperature=0,
                                              messages=[
                                                  {"role": "system", "content": prompt},
                                                  {"role": "user", "content": "Generate the SPARQL query."}
                                              ])

    # Extracting the SPARQL query from the response
    sparql_query = response.choices[0].message.content.strip()

    # Remove any non-standard prefixes like ```sparql
    sparql_query = sparql_query.replace("```sparql", "").replace("```", "").strip()

    return sparql_query

@timed_function
def match_entities(entities, vector_db):
    matched_entities = []
    for entity in entities:
        if entity:  # Check if the entity string is not empty
            # Split the entity string and use only the part after the ': '
            entity_attributes = entity.split(': ', 1)[1] if ': ' in entity else entity
            matches = vector_db.similarity_search(entity_attributes)
            if matches:  # Check if there is at least one match
                print()
                matched_entities.append(matches[0])  # Append only the top result
    return matched_entities

@timed_function
def match_relations(relations, vector_db):
    matched_relations = []
    for relation in relations:
        if relation:  # Check if the relation string is not empty
            matches = vector_db.similarity_search(relation)
            if matches:  # Check if there is at least one match
                matched_relations.append(matches[0])  # Append only the top result
                matched_relations.append(matches[1]) 
    return matched_relations

@timed_function
def generate_sparql_query(question):
    # Extract entities and relations
    entities, relations = extract_entities_relations(question)

    print("Extracted Entities: ", entities)
    print("Extracted Relations: ", relations)

    # Match the candidate entities and relations using vector similarity
    matched_entities = match_entities(entities, vector_db_entities)
    matched_relations = match_relations(relations, vector_db_relations)

    print("Matched Entities: ", [str(entity) for entity in matched_entities])
    print("Matched Relations: ", [str(relation) for relation in matched_relations])

    # Construct the SPARQL query using OpenAI
    sparql_query = construct_sparql_query_openai(question, matched_entities, matched_relations)
    print(f"\nConstructed SPARQL Query: \n{sparql_query}")
    # Initialize DKG
    ot_node_hostname = os.getenv("OT_NODE_HOSTNAME") + ":8900"
    node_provider = NodeHTTPProvider(ot_node_hostname)
    blockchain_provider = BlockchainProvider(
            os.getenv("RPC_ENDPOINT"), 
            os.getenv("WALLET_PRIVATE_KEY")
        )

    # initialize the DKG client on OriginTrail DKG
    dkg = DKG(node_provider, blockchain_provider)

    query_graph_result = dkg.graph.query(
    sparql_query,
    repository="privateCurrent",
    )


    print(query_graph_result)

    return query_graph_result



if __name__ == "__main__":
    import sys

    # Get the question from command line argument
    question = sys.argv[1]
    result = generate_sparql_query(question)
    print(result)