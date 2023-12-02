from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from utils.generic_cypher import Neo4jGPTQuery
import neo4j
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from neo4j import GraphDatabase
from langchain.chains import GraphQAChain
from langchain.indexes.graph import NetworkxEntityGraph

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
                openai_key=None,
                pinecone_api_key=None,
                pinecone_env_name=None,
                pinecone_index_name=None,):
        
        self.neo4j_url=neo4j_url
        self.neo4j_user=neo4j_user
        self.neo4j_password=neo4j_password
        self.openai_key=openai_key
        self.pinecone_api_key=pinecone_api_key
        self.pinecone_env_name=pinecone_env_name
        self.pinecone_index_name=pinecone_index_name
        self.embeddings = OpenAIEmbeddings()
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

        pinecone.init(api_key=self.pinecone_api_key,environment=self.pinecone_env_name)
        self.index = pinecone.Index(self.pinecone_index_name)

    # Experiment 2: Generic Cypher
    def generic_cypher(self, query):
        self.db = Neo4jGPTQuery(
                                url=self.neo4j_url,
                                user=self.neo4j_user,
                                password=self.neo4j_password,
                                openai_api_key=self.openai_key)
        
        response = self.db.run(query)
        response =list_to_string(response)
        return response
    
    # Experiment 3: Graph Cypher QA
    def graph_cypher_qa(self, question):
        graph = Neo4jGraph(
                            url=self.neo4j_url,
                            username=self.neo4j_user,
                            password=self.neo4j_password)
        
        chain = GraphCypherQAChain.from_llm(ChatOpenAI(temperature=0), graph=graph, verbose=True,)
        response = chain.run(question)
        return response
    
    # Experment 4: QAChian-graph-Pinecone
    def graph_qa_graph_pinecone(self, question, my_namespace="graph_02", text_key="context", topK=10,):
        vectorstore = Pinecone(self.index , self.embeddings.embed_query, text_key, namespace=my_namespace)
        docs = vectorstore.similarity_search(question, k=topK)
        chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]
    
    # Experment 4: QAChian-pdf-Pinecone
    def graph_qa_pdf_pinecone(self, question, my_namespace="unilever", text_key="text", topK=5):
        vectorstore = Pinecone(self.index , self.embeddings.embed_query, text_key, namespace=my_namespace)
        docs = vectorstore.similarity_search(question, k=topK)
        chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]
    

    # Experment 5: GraphQAChain with Knowledge Triples
    def graph_qa_knowledge_triples(self, question, graph_pth="ps-graph.gml"):
        loaded_graph = NetworkxEntityGraph.from_gml(graph_pth)
        chain = GraphQAChain.from_llm(OpenAI(temperature=0), graph=loaded_graph, verbose=True)
        response = chain.run(question)
        return response
    
    # Optimissed version of Cyher Query
    def optimised_cyher(self, question, my_namespace="graph", text_key="name", topK=20):
        vectorstore = Pinecone(self.index , self.embeddings.embed_query, text_key, namespace=my_namespace)

        docs = vectorstore.similarity_search(question, k=topK)

        # Get the related node names, node types and edges
        related_node_names = ""
        related_node_types = ""
        related_edges = ""
        for d in docs: 
            if d.metadata["info_type"] == "node_names":
                related_node_names+=d.page_content + ", "
            
            if d.metadata["info_type"] == "node_types":
                related_node_types+=d.page_content + ", "
            
            if d.metadata["info_type"] == "edges":
                related_edges+=d.page_content + ", "

        # Build additional hint
        additional_hint = f"""Hint: Please refer the names of nodes:{related_node_names} or the labels of nodes: {related_node_types} or the edges type: {related_edges} if necessary"""

        # Enquiry the database as graph cypher QA
        graph = Neo4jGraph(
                            url=self.neo4j_url,
                            username=self.neo4j_user,
                            password=self.neo4j_password)
        
        chain = GraphCypherQAChain.from_llm(ChatOpenAI(temperature=0), graph=graph, verbose=True,)
        response = chain.run(question + additional_hint)

        return response