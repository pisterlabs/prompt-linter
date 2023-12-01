from dotenv import load_dotenv
import os
import pandas as pd
import json
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.embeddings import OpenAIEmbeddings
from sqlalchemy import create_engine
import langchain
# Load environnement variables
load_dotenv()  

URI = os.environ["NEO4J_INSTANCE01_URI"]
USER = os.environ["NEO4J_INSTANCE01_USER"]
PWD = os.environ["NEO4J_INSTANCE01_KEY"]

graph = Neo4jGraph(url=URI, username=USER, password=os.environ["NEO4J_INSTANCE01_KEY"])
llm = ChatOpenAI(temperature=0,model='gpt-4')
embedding = OpenAIEmbeddings(model='text-embedding-ada-002')
connection = create_engine(os.environ.get('DATABASE_URL'))
chat = langchain.chat_models.ChatOpenAI(temperature=0.2, model_name='gpt-4',)


def get_basic_chat_chain(system_template: str, user_template: str) -> langchain.LLMChain:
    """
    Returns a langchain chain given an user and chat inputs

    Parameters:
        system_template (str): The system template
        user_template (str): The user template

    Returns:
        langchain.LLMChain: The langchain chain
    """

    system_prompt = langchain.prompts.SystemMessagePromptTemplate.from_template(
        system_template)

    user_prompt = langchain.prompts.HumanMessagePromptTemplate.from_template(
        user_template)

    chat_prompt = langchain.prompts.ChatPromptTemplate.from_messages([
        system_prompt,
        user_prompt
    ])

    return langchain.LLMChain(llm=chat, prompt=chat_prompt)
def search_similar_nodes(text_list : list):
    nodes = []
    for text in text_list:
        text_embedding = embedding.embed_query(text)
        vector_string = str(text_embedding)
        sql_query = "SELECT content FROM nodes_embeddings ORDER BY embedding <-> '"+ vector_string +"' LIMIT 1;"
        query_result = pd.read_sql(sql_query,connection)
        nodes += query_result.content.to_list()
    return nodes

neo4j_search_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    top_k=100,
    return_direct=True
)

sys_temp = """
You will be asked a question about a scientific article.
Your job is to identify which information do you need to answer the question.

You have to return a JSON object. Here is an example:
{{
    'initial_question': 'What is the article about ?',
    'analisis' : 'In order to answer this question I need to retrieve an Abstract or a Summary of the article',
    'informations_needed': ['Abstract','Summary']
}}

Another example:
{{
    'initial_question': 'How does the moon exploration contribute to the humanity?',
    'analisis' : 'In order to answer this question I information about moon exploration and humanity',
    'informations_needed': ['moon exploration','humanity']
}}

# Rules
You return maximum two informations needed. Not more than two so chose them wisely.
Just return the JSON Object, nothing more.
No prefacing text. Your answer have to begin with a left curly bracket and end with a right curly bracket.
"""

user_temp = """
## Input
{question}
"""
extract_concepts_chain = get_basic_chat_chain(sys_temp,user_temp)

sys_temp = """
You will be given as input :
- a question about a scientific article 
- the text representating of a knowledge graph containing usefull information to answer the question

Your job is answer as best as you can to the question, using only the information provided in the knowledge graph.
Be short and concise. 
"""

user_temp = """
## Question
{question}

## Knowledge graph
{graph}
"""
final_answer_chain = get_basic_chat_chain(sys_temp,user_temp)

def answer_question(question):
    # First step is to identify the information I need to answer the question
    info_needed = extract_concepts_chain.run(question = question)
    print(info_needed)
    info_needed = json.loads(info_needed.replace("'",'"'))

    # Get the nodes 
    interesting_nodes = search_similar_nodes(info_needed['informations_needed'])
    interesting_nodes = list(np.unique(interesting_nodes))
    print("interesting nodes: " + str(interesting_nodes))

    # Get the subgraph
    query = "Return the neighborhood subgraph of size 2 of nodes " + str(interesting_nodes)
    subgraph = neo4j_search_chain.run(query)
    subgraph= str(subgraph).replace("{","{{").replace("}","}}")

    # Get final answer
    final_answer = final_answer_chain.run(question = question, graph = subgraph)

    return final_answer