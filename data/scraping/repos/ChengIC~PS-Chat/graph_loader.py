
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from neo4j import GraphDatabase
import itertools

class ingest_graph:
    def __init__(self, namespace, neo4j_url, neo4j_user, neo4j_password,
                pinecone_api_key, pinecone_env_name, pinecone_index_name):
        
        self.neo4j_url=neo4j_url
        self.neo4j_user=neo4j_user
        self.neo4j_password=neo4j_password

        self.namespace=namespace
        self.pinecone_api_key=pinecone_api_key
        self.pinecone_env_name=pinecone_env_name
        self.pinecone_index_name=pinecone_index_name

        # initialize pinecone
        self.pinecone = pinecone.init(
                        api_key=pinecone_api_key,  # find at app.pinecone.io
                        environment=pinecone_env_name  # next to api key in console
                    )
    
    def upload_graph_to_pinecone(self, source_name="neo4j-graph"):
        embeddings = OpenAIEmbeddings()

        print ("Start loading graph...")
        index = pinecone.Index(self.pinecone_index_name)
        index.delete(deleteAll='true', namespace=self.namespace)
        print ("clearing pinecone index if namespace exists...")
        
        driver = GraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_user, self.neo4j_password))
        # Divide the graph into trunks
        def get_pair_nodes (tx):
            pairs_of_nodes = []
            for record in tx.run("MATCH (a)-[r]->(b) RETURN labels(a), a.name, type(r), labels(b), b.name"):
                pair_node = {
                    "start_node_name": record["a.name"],
                    "start_node_label":record["labels(a)"][0],
                    "end_node_name": record["b.name"],
                    "end_node_label":record["labels(b)"][0],
                    "edge": record["type(r)"]
                }
                pairs_of_nodes.append(pair_node)
                print(record["labels(a)"][0], record["a.name"], record["type(r)"], record["labels(b)"][0], record["b.name"])
            return pairs_of_nodes
        
        with driver.session() as session:
            pairs_of_nodes = session.execute_read(get_pair_nodes)
        driver.close()
        
        def get_node_vectors(set_data, info_type):
            node_vector_list = []
            for n in set_data:
                try:
                    embedded_value = embeddings.embed_documents([n])
                    node_vector_list.append({
                        "value": embedded_value,
                        "meta_data": {
                            "source": source_name,
                            "info_type": info_type,
                            "text": n
                        }
                    })
                except Exception as e:
                    print(f"Error embedding node {n}: {e}")
                    continue
            return node_vector_list

        node_names = {p_n["start_node_name"] for p_n in pairs_of_nodes}.union(
                    {p_n["end_node_name"] for p_n in pairs_of_nodes})
        node_types = {p_n["start_node_label"] for p_n in pairs_of_nodes}.union(
                    {p_n["end_node_label"] for p_n in pairs_of_nodes})
        edge_types = {p_n["edge"] for p_n in pairs_of_nodes}

        node_vectors = get_node_vectors(node_names, "node_names") + \
                    get_node_vectors(node_types, "node_types") + \
                    get_node_vectors(edge_types, "edges")
        
        pinecone_vectors = []
        for idx, n_v in enumerate(node_vectors):
            pinecone_vectors.append((str(idx), n_v["value"], n_v["meta_data"]))

        # Upload the vectors to Pinecone with batches
        def chunks(iterable, batch_size=100):
            """A helper function to break an iterable into chunks of size batch_size."""
            it = iter(iterable)
            chunk = tuple(itertools.islice(it, batch_size))
            while chunk:
                yield chunk
                chunk = tuple(itertools.islice(it, batch_size))
        for vec in chunks(pinecone_vectors, batch_size=10):
            index.upsert(vectors=vec,namespace=self.namespace)

        # upsert_response = index.upsert(vectors=pinecone_vectors, namespace=self.namespace)
        print ("Finished loading graph...")
