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

openai_keys_file = os.path.join(os.getcwd(), "keys/openai_keys.txt")
with open(openai_keys_file, "r") as f:
    keys = f.read()
keys = keys.strip().split('\n')
os.environ["OPENAI_API_KEY"] = keys[0]

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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

entity_names_query = """
SELECT entities.entity_id, attribute_value "name" FROM entities
    JOIN entity_attributes_str ON entities.entity_id = entity_attributes_str.entity_id
    WHERE attribute_name = 'name';
"""

cur.execute(entity_names_query)
entity_names = ", ".join([row[1] for row in cur.fetchall()])
cur.close()

ENTITY_SELECT_TEMPLATE = f"""
A question is provided below. Given the question, select up to {{max_keywords}} entity names
from the provided entity names. Focus on selecting the entities that might be relevant to the
question.

Provide keywords in the following comma-separated format: 'KEYWORDS: <selected entity names>'

Here is an example:
```
---------------------
PROVIDED ENTITY NAMES: robot, map, point, pose, region, door, orange, the_living_room, stephanie_bedroom, beige_pen, spoon, espn, the_living_room_tv, light, apple, stephanie_bedroom_tv
---------------------
QUESTION: What TVs are there and where are they playing?
---------------------
KEYWORDS: the_living_room_tv, espn, stephanie_bedroom_tv
---------------------
```

Here is the real problem:
```
---------------------
PROVIDED ENTITY NAMES: {entity_names}
---------------------
QUESTION: {{question}}
---------------------
"""

ENTITY_SELECT_PROMPT = PromptTemplate(
    ENTITY_SELECT_TEMPLATE,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
    graph_traversal_depth=3,
    max_knowledge_sequence=100,
    entity_extract_template=ENTITY_SELECT_PROMPT
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever, service_context=service_context
)

# response = query_engine.query("Where is the sink?")
# response = query_engine.query("What fruits are in the household and where are they?")
# response = query_engine.query("Where can the robot go to find an apple?")
# response = query_engine.query("What type of objects can you find on the desk?")
# response = query_engine.query("How tall is the highest level of pantry?")
# response = query_engine.query("How wide is the desk?")
response = query_engine.query("What objects are in the living room?")
print(response)

conn.close()