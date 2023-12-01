from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from utils.generic_cypher import Neo4jGPTQuery
import neo4j
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from neo4j import GraphDatabase
from langchain.callbacks import get_openai_callback
from utils.count_tokens import log_token_details_to_file


def list_to_string(lst):
    result = ''
    for item in lst:
        if isinstance(item, list):
            result += list_to_string(item) + '\n'
        elif isinstance(item, neo4j.graph.Node):
            result += str(dict(item)) + ', '
        elif isinstance(item, neo4j.graph.Relationship):
            result += f"({dict(item.start_node)})-[:{item.type}]->({dict(item.end_node)})" + ', '
        else:
            result += str(item) + ', '
    return result


class QueryGraph():
    def __init__(self, neo4j_url=None, neo4j_user=None, neo4j_password=None, 
                openai_key=None, model_version="gpt-3.5-turbo-16k"):
        
        self.neo4j_url=neo4j_url
        self.neo4j_user=neo4j_user
        self.neo4j_password=neo4j_password
        self.openai_key=openai_key
        self.embeddings = OpenAIEmbeddings()
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
        self.model_version = model_version

    # Experiment 2: Generic Cypher
    def generic_cypher(self, query):
        self.db = Neo4jGPTQuery(
                                url=self.neo4j_url,
                                user=self.neo4j_user,
                                password=self.neo4j_password,
                                openai_api_key=self.openai_key,
                                model_version=self.model_version)
        
        response = self.db.run(query)
        print (response)
        response =list_to_string(response)
        
        return response
    
    # Experiment 3: Graph Cypher QA
    def graph_cypher_qa(self, question):
        graph = Neo4jGraph(
                            url=self.neo4j_url,
                            username=self.neo4j_user,
                            password=self.neo4j_password)
        
        chain = GraphCypherQAChain.from_llm(ChatOpenAI(model=self.model_version, temperature=0), graph=graph, verbose=True,)
        response = chain.run(question)
        print (response)
        return response
    
    # Optimissed version of Cypher Query
    def optimised_cypher(self, question, pinecone_api_key,pinecone_env_name,pinecone_index_name,
                        my_namespace="graph", text_key="text", topK=20):
        
        pinecone.init(api_key=pinecone_api_key,environment=pinecone_env_name)
        index = pinecone.Index(pinecone_index_name)
        vectorstore = Pinecone(index, self.embeddings.embed_query, text_key, namespace=my_namespace)
        docs = vectorstore.similarity_search_with_score(question, k=topK)
        print ("Finding associated information from graph in the pinecone index...")
        print (docs)
        print ("**************************************")

        # Get the related node names, node types and edges
        node_names_info, node_types_info, edges_info= "", "", ""
        len_node_names, len_node_types, len_edges = 0, 0, 0
        for doc in docs:
            d = doc[0]
            if d.metadata["info_type"] == "node_names":
                len_node_names+=1
                node_names_info += f"({len_node_names}) " + d.page_content + "; "
            
            if d.metadata["info_type"] == "node_types":
                len_node_types+=1
                node_types_info += f"({len_node_types}) " + d.page_content + "; "
            
            if d.metadata["info_type"] == "edges":
                len_edges+=1
                edges_info += f"({len_edges}) " + d.page_content + "; "

        # Build additional hint
        if len_node_names!=0 or len_node_types!= 0 or len_edges!=0:
            additional_hint = f""" (I also used KNN and Pinecone to find the relevant nodes names, nodes labels, and edges relating to the questions, which might be helpful for you. 
                                    Noted that these additional hints might be noisy and misleading. 
                                    For example, 'AA_Optimisation' is different from 'BB_Optimisation'. 'AA Optimisation' is also different from 'BB Optimization'.
                                    Be careful when dealing with the following hints by the guidance."""
            
            if len_node_names != 0:
                additional_hint += f"""Hint: Relevant nodes' names: {node_names_info}. """

            if len_node_types != 0:
                additional_hint += f"""Relevant nodes' labels: {node_types_info}. """

            if len_edges != 0:
                additional_hint += f"""Relevant edges: {edges_info}."""

            question = question + additional_hint + ")"

            print ("Using the following additional hint as")
            print (question)
            print ("**************************************")

        # Enquiry the database as graph cypher QA
        graph = Neo4jGraph(
                            url=self.neo4j_url,
                            username=self.neo4j_user,
                            password=self.neo4j_password)
        
        chain = GraphCypherQAChain.from_llm(ChatOpenAI(model=self.model_version,temperature=0), graph=graph, verbose=True,)
        with get_openai_callback() as cb:
            response = chain.run(question)

            # log token details
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            log_token_details_to_file(cb.prompt_tokens, cb.completion_tokens, self.model_version)

        return response