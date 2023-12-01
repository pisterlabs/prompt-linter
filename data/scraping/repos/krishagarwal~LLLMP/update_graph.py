import os
import logging
import sys
from llama_index.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.prompts import PromptTemplate, PromptType
from llama_index import PromptTemplate, ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import KnowledgeGraphRAGRetriever
import re

# set API key
openai_keys_file = os.path.join(os.getcwd(), "keys/openai_keys.txt")
with open(openai_keys_file, "r") as f:
    keys = f.read()
keys = keys.strip().split('\n')
os.environ["OPENAI_API_KEY"] = keys[0]

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)



graph_name = "knowledge_graph"
from age import AgeGraphStore
graph_store = AgeGraphStore(
    dbname="knowledge_base",
    user="postgres",
    password="password",
    host="localhost",
    port=5432,
    graph_name=graph_name,
    node_label="entity"
)
conn = graph_store.client()

# Create a new database session and return a new instance of the connection class
cur = conn.cursor()
cur.execute(f"LOAD 'age'")
cur.execute(f"SET search_path = ag_catalog, '$user', public;")



# get all the entity_names (used for entity selection)
entity_names_query = """
SELECT entities.entity_id, attribute_value "name" FROM entities
    JOIN entity_attributes_str ON entities.entity_id = entity_attributes_str.entity_id
    WHERE attribute_name = 'name';
"""
cur.execute(entity_names_query)
entity_names = ", ".join([row[1] for row in cur.fetchall()])



# read prompt template from file and format
def get_prompt_template(filename: str, **kwargs) -> str:
    with open(os.path.join(os.path.dirname(__file__), filename), "r") as f:
        contents = f.read()
    if not kwargs:
        return contents
    return contents.format(**kwargs)

# load in all default prompts
ENTITY_SELECT_TEMPLATE = get_prompt_template("entity_select_prompt.txt", entity_names=entity_names)
ENTITY_SELECT_PROMPT = PromptTemplate(
    ENTITY_SELECT_TEMPLATE,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)
TRIPLET_FILTER_PROMPT = get_prompt_template("triplet_filter_prompt.txt")
TRIPLET_UPDATE_PROMPT = get_prompt_template("triplet_update_prompt.txt")



llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)
storage_context = StorageContext.from_defaults(graph_store=graph_store)
graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
    graph_traversal_depth=2,
    max_knowledge_sequence=100,
    entity_extract_template=ENTITY_SELECT_PROMPT
)

# format triplets from AGE query output
def postprocess_triplet(triplet: str) -> str:
    components = [re.sub(r'[^a-zA-Z0-9_]', '', component) for component in triplet.split(", ")]
    return " -> ".join(components)

def process_state_change(state_change: str, out_file_name: str) -> None:
    output = "------------------------------------------------\n"
    output += f"STATE CHANGE: {state_change}\n"

    # retrieve relevant triplets with llama-index (but prevent llama-index printing)
    with open(os.devnull,"w") as devNull:
        orig = sys.stdout
        sys.stdout = devNull
        context_nodes = graph_rag_retriever.retrieve(state_change)
        sys.stdout = orig
    context_str = context_nodes[0].text if len(context_nodes) > 0 else "None"
    triplets = [postprocess_triplet(triplet) for triplet in context_str.split('\n')[2:]]
    triplet_str = '\n'.join(triplets)

    output += "------------------------------------------------\n"
    output += "EXTRACTED TRIPLETS:\n\n"
    output += triplet_str + "\n"

    # filter out irrelevant triplets using LLM directly
    filtered_triplet_str = llm.complete(TRIPLET_FILTER_PROMPT.format(state_change=state_change, triplet_str=triplet_str)).text
    output += "------------------------------------------------\n"
    output += "FILTERED TRIPLETS:\n\n"
    output += filtered_triplet_str + "\n"

    # query LLM to update triplets (remove existing and add new)
    triplet_updates = llm.complete(TRIPLET_UPDATE_PROMPT.format(state_change=state_change, filtered_triplet_str=filtered_triplet_str)).text
    output += "------------------------------------------------\n"
    output += "TRIPLETS TO ADD AND REMOVE\n\n"
    output += triplet_updates + "\n"
    output += "------------------------------------------------\n"

    # output the results
    with open(out_file_name, "w") as f:
        f.write(output)

    # process the changes and commit to knowledge graph
    update_lines = triplet_updates.split('\n')
    remove_idx = update_lines.index("REMOVE:")
    add_idx = update_lines.index("ADD:")
    remove = update_lines[remove_idx + 1 : add_idx]
    add = update_lines[add_idx + 1:]

    # delete triplets from graph
    for triplet_str in remove:
        triplet = triplet_str.split(" -> ")
        if len(triplet) == 3:
            cur.execute(f"SELECT * FROM cypher('{graph_name}', $$MATCH (u {{name: '{triplet[0]}'}})-[e:{triplet[1]}]->(v {{name: '{triplet[2]}'}}) DELETE e$$) as (e agtype);")

    # add new triplets to graph
    for triplet_str in add:
        triplet = triplet_str.split(" -> ")
        if len(triplet) == 3:
            cur.execute(f"SELECT * FROM cypher('{graph_name}', $$MATCH (u {{name: '{triplet[0]}'}}), (v {{name: '{triplet[2]}'}}) CREATE (u)-[e:{triplet[1]}]->(v) RETURN e$$) as (e agtype);")
    
    conn.commit()
    print("Completed update:", state_change)


# set of state changes
state_changes = ["I ate the pineapple.", "I moved the knife to the living room table."]

for i, state_change in enumerate(state_changes):
    process_state_change(state_change, f"knowledge/result{i}.txt")

cur.close()
conn.close()