from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.graphs import MemgraphGraph

query: str = ""

graph = MemgraphGraph(
    url="bolt://127.0.0.1:7687", username="brettb", password="password"
)

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
If the user asks about the Red Wedding, that is the episode called The Rains of Castamere.
If the user asks who was killed by hanging, you should return the characters that were killed with a relationship property of method equal to noose.
The method property is only available for the KILLED relationship.
If you use the method property of the KILLED relationship, make that part of the Cypher statement case insensitive.

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authorative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Information:
{context}

Question: {question}
Helpful Answer:"""

CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0, model="gpt-4"),
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    qa_prompt=CYPHER_QA_PROMPT,
    top_k=50,
)

###############################################################################
# ###### queries that leverage nodes and relationships
# query = "How many seasons were there?"
# query = "Who is Jon Snow allied with?"
# query = "Show me the characters with allegiance to the white walkers."
# query = "Which characters had the most allegiances?"
# query = (
#    "Which characters had the most allegiances, and show me what they were in a table?"
# )
# query = "Where did Jon Snow die?"
# query = (
#    "Which episode had the most deaths of characters allied with the House Targaryen?"
# )
query = (
    "Which episode had the most deaths of characters allied with the House Stark,"
    " and who were they?"
)
# query = "Who were the victims in the episode called The Long Night?"
# query = "Who died in the red wedding?"

###############################################################################
###### queries that leverage the method property of the KILLED relationship
# query = "Which character was killed by a Shadow Demon?"
# query = "Which characters were killed by hanging?"
# query = "What was used to kill Jon Snow?"
# query = "Which characters were killed by poison?"

try:
    results = chain.run(query)
    print(results)
except Exception as e:
    print(e)

