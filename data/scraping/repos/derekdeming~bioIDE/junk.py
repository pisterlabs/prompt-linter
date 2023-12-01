import concurrent.futures
from typing import Dict, List
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.node_parser import SimpleNodeParser
from llama_index.data_structs import Node
from llama_index.schema import Document
from pathlib import Path
from llama_index import download_loader
from llama_index import Document
import ray
from ray.data import ActorPoolStrategy # to use Actors for parallelization -- map_batches() method


ray.init()

# Step 1: Logic for parsing the files into llama_index documents - in our case it will be in formats of json (subject to change)
def process_paper_data(paper_json):
    '''
    This function takes in a json object and returns a llama_index Document object
    
    :param paper_json: a json object containing the data for a single paper
    :return: a llama_index Document object
    '''
    paper_data = {}

    paper_data['doi'] = paper_json['doi']
    paper_data['title'] = paper_json['title']
    paper_data['authors'] = paper_json['authors'].split('; ')
    paper_data['date'] = paper_json['date']
    paper_data['category'] = paper_json['category']
    paper_data['abstract'] = paper_json['abstract']

    # combining all relevant data from the papers into a single string
    paper_text = f"Title: {paper_data['title']}\nAuthors: {', '.join(paper_data['authors'])}\nDate: {paper_data['date']}\nCategory: {paper_data['category']}\nAbstract: {paper_data['abstract']}"

    # instantiate a Document object from llama_index.schema -- requires a doc_id and text and we will use the doi as the doc_id
    document = Document(doc_id=paper_data['doi'], text=paper_text)

    return document

def convert_documents_into_nodes(documents: Dict[str, Document]) -> Dict[str, Node]:
    '''
    This function takes in a dictionary of documents and returns a dictionary of nodes

    :param documents: a dictionary of documents
    :return: a dictionary of nodes
    '''
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents([document for document in documents.values()])
    return [{"node": node} for node in nodes]

@ray.remote
def process_jsons_parallel(jsons: List[Dict]):
    '''
    This function takes in a list of json objects and returns a list of nodes. Idea is to parallelize the processing of the jsons and the creation of the documents
    
    :param jsons: a list of json objects
    :return: a list of nodes
    '''
    document_results = [process_paper_data(json) for json in jsons]
    nodes = convert_documents_into_nodes({doc.doc_id: doc for doc in document_results})
    return nodes
    
# Embed each node using a local embedding model 
@ray.remote
class EmbedNodes:
    def __init__(self):
        '''
        Use all-mpnet-base-v2 Sentence_transformer.
        This is the default embedding model for LlamaIndex/Langchain.

        Use GPU for embedding and specify a large enough batch size to maximize GPU utilization.
        Remove the "device": "cuda" to use CPU instead.
        '''
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100}
            )
    
    def __call__(self, node_batch: Dict[str, List[Node]]) -> Dict[str, List[Node]]:
        nodes = node_batch["node"]
        text = [node.text for node in nodes]
        embeddings = self.embedding_model.embed_documents(text)
        assert len(nodes) == len(embeddings)

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return {"embedded_nodes": nodes}

def create_ray_dataset_pipeline(jsons: List[Dict]):
    # Create the Ray Dataset pipeline
    ds = ray.data.from_items(jsons)
    
    # Initialize the actor
    embed_nodes_actor = EmbedNodes.remote()

    def embed_nodes(node_batch):
        return ray.get(embed_nodes_actor.__call__.remote(node_batch))

    # Use `map_batches` to specify a batch size to maximize GPU utilization.
    embedded_nodes = ds.map_batches(
        embed_nodes, 
        batch_size=100,
        # There are 4 GPUs in the cluster. Each actor uses 1 GPU. So we want 4 total actors.
        # Set the size of the ActorPool to the number of GPUs in the cluster.
        compute=ActorPoolStrategy(size=1), 
        )
    
    # Step 5: Trigger execution and collect all the embedded nodes.
    ray_docs_nodes = []
    for row in embedded_nodes.iter_rows():
        node = row["embedded_nodes"]
        assert node.embedding is not None
        ray_docs_nodes.append(node)

    return ray_docs_nodes

# HANDLING LOCAL FILES - NOT NEEDED FOR NOW
# Step 0: Logic for loading and parsing the files into llama_index documents.
# UnstructuredReader = download_loader("UnstructuredReader")
# loader = UnstructuredReader()

# def load_and_parse_files(file_row: Dict[str, Path]) -> Dict[str, Document]:
#     documents = []
#     file = file_row["path"]
#     if file.is_dir():
#         return []
#     # Skip all non-html files like png, jpg, etc.
#     if file.suffix.lower() == ".html":
#         loaded_doc = loader.load_data(file=file, split_documents=False)
#         loaded_doc[0].extra_info = {"path": str(file)}
#         documents.extend(loaded_doc)
#     return [{"doc": doc} for doc in documents]





def main():
    db_manager = DatabaseManager()

    response  = db_manager.fetch("biorxiv", "fetch_details", server="biorxiv", interval="2021-06-01/2021-06-05")
    papers = response.json()['collection']

    # Process each paper separately and store the resulting Documents in a dictionary
    documents = {paper['doi']: process_paper_data(paper) for paper in papers}

    # Convert the dictionary of Documents into Nodes
    nodes = convert_documents_into_nodes(documents)
    print(nodes)

    # Just printing out the first three for testing purposes
    for i, node in enumerate(nodes):
        print(node)
        print("\n\n")
        if i >= 2:  
            break





# Step 1: Logic for parsing the files into llama_index documents - in our case it will be in formats of json (subject to change)
def process_paper_data(paper_json):
    '''
    This function takes in a json object and returns a llama_index Document object
    
    :param paper_json: a json object containing the data for a single paper
    :return: a llama_index Document object
    '''
    paper_data = {}

    paper_data['doi'] = paper_json['doi']
    paper_data['title'] = paper_json['title']
    paper_data['authors'] = paper_json['authors'].split('; ')
    paper_data['date'] = paper_json['date']
    paper_data['category'] = paper_json['category']
    paper_data['abstract'] = paper_json['abstract']

    # combining all relevant data from the papers into a single string
    paper_text = f"Title: {paper_data['title']}\nAuthors: {', '.join(paper_data['authors'])}\nDate: {paper_data['date']}\nCategory: {paper_data['category']}\nAbstract: {paper_data['abstract']}"

    # instantiate a Document object from llama_index.schema -- requires a doc_id and text and we will use the doi as the doc_id
    document = Document(doc_id=paper_data['doi'], text=paper_text)

    return document





# 33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333


# from db_wrapper import BiorxivDatabase, EnsemblDatabase, GeoDatabase, UniProtDatabase
# from utils.vector_index import EmbedNodes
from typing import List
from utils.parser import process_paper_data, convert_documents_into_nodes, EmbedNodes
from database_manager import DatabaseManager
from llama_index import GPTVectorStoreIndex
import ray
from ray.data import Dataset



def main():
    # Instantiate EmbedNodes object
    embedder = EmbedNodes()

    db_manager = DatabaseManager()

    response = db_manager.fetch("biorxiv", "fetch_details", server="biorxiv", interval="2021-06-01/2021-06-05")
    papers = response.json()['collection']

    # Process each paper separately and store the resulting Documents in a dictionary
    documents = {paper['doi']: process_paper_data(paper) for paper in papers}

    # Convert the dictionary of Documents into Nodes
    nodes_dict = convert_documents_into_nodes(documents)

    # Convert the nodes_dict into a list of nodes
    nodes = [node_dict["node"] for node_dict in nodes_dict]

    # Convert nodes into a Ray dataset
    nodes_ds = Dataset.from_pandas(nodes)

    # Embed nodes using Ray
    embedded_nodes_ds = nodes_ds.map_batches(
        embedder, 
        batch_size=100, 
        num_gpus=1,
        compute=ray.data.ActorPoolStrategy(size=1), 
    )

    # Collect all the embedded nodes.
    bio_docs_nodes = []
    for row in embedded_nodes_ds.iter_rows():
        node = row["embedded_nodes"]
        assert node.embedding is not None
        bio_docs_nodes.append(node)

    # Store the embedded nodes in a local vector store, and persist to disk.
    print("Storing BioRxiv Document embeddings in vector index.")
    bio_docs_index = GPTVectorStoreIndex(nodes=bio_docs_nodes)
    bio_docs_index.storage_context.persist(persist_dir="/tmp/bio_docs_index")










    # from db_wrapper import BiorxivDatabase, EnsemblDatabase, GeoDatabase, UniProtDatabase
from typing import List
from pathlib import Path
from utils.parser import convert_documents_into_nodes, EmbedNodes, load_and_parse_json
from database_manager import DatabaseManager
from llama_index import GPTVectorStoreIndex
import ray
from ray.data import Dataset
from ray.data import ActorPoolStrategy
import json

def main():
    db_manager = DatabaseManager()

    response  = db_manager.fetch("biorxiv", "fetch_details", server="biorxiv", interval="2021-06-01/2021-06-05")
    papers = response.json()['collection']

    # Process each paper separately and store the resulting Documents in a dictionary
    documents = {paper['doi']: load_and_parse_json(paper)['doc'] for paper in papers}

    # Convert the dictionary of Documents into Nodes
    biorxiv_nodes = convert_documents_into_nodes(list(documents.values()))

    # Create the Ray Dataset pipeline
    biorxiv_ds = ray.data.from_items([node['node'] for node in biorxiv_nodes], parallelism=20)

    # Use `map_batches` to specify a batch size to maximize GPU utilization.
    # We define `EmbedNodes` as a class instead of a function so we only initialize the embedding model once. 
    #   This state can be reused for multiple batches.
    embedded_nodes = biorxiv_ds.map_batches(
        EmbedNodes, 
        batch_size=1, 
        # Use 1 GPU per actor.
        num_cpus=1,
        num_gpus=None,
        # There are 4 GPUs in the cluster. Each actor uses 1 GPU. So we want 4 total actors.
        # Set the size of the ActorPool to the number of GPUs in the cluster.
        compute=ActorPoolStrategy(size=15), 
    )

    # Trigger execution and collect all the embedded nodes.
    biorxiv_docs_nodes = []
    for row in embedded_nodes.iter_rows():
        node = row["embedded_nodes"]
        assert node.embedding is not None
        biorxiv_docs_nodes.append(node)

    # Store the embedded nodes in a local vector store, and persist to disk.
    print("Storing Ray Documentation embeddings in vector index.")

    biorxiv_docs_index = GPTVectorStoreIndex(nodes=biorxiv_docs_nodes)
    biorxiv_docs_index.storage_context.persist(persist_dir="/tmp/biorxiv_docs_index")



    # # Get the paths for the locally downloaded documentation.
    # all_docs_gen = Path("./bio_papers").rglob("*")
    # all_docs = [{"path": doc.resolve()} for doc in all_docs_gen]

    # # Create the Ray Dataset pipeline
    # local_ds = ray.data.from_items(all_docs)
    # loaded_docs = local_ds.map_batches(load_and_parse_files)
    # bio_local_paper_nodes = loaded_docs.map_batches(convert_documents_into_nodes)
    # embedded_nodes = bio_local_paper_nodes.map_batches(
    #     EmbedNodes, 
    #     batch_size=100, 
    #     num_gpus=1,
    #     compute=ActorPoolStrategy(size=1), 
    #     )


    # # Step 5: Trigger execution and collect all the embedded nodes.
    # bio_local_docs_nodes = []
    # for row in embedded_nodes.iter_rows():
    #     node = row["embedded_nodes"]
    #     assert node.embedding is not None
    #     bio_local_docs_nodes.append(node)

    # # Step 6: Store the embedded nodes in a local vector store, and persist to disk.
    # print("Storing Ray Documentation embeddings in vector index.")

    # bio_local_docs_index = GPTVectorStoreIndex(nodes=bio_local_docs_nodes)
    # bio_local_docs_index.storage_context.persist(persist_dir="/tmp/bio_local_docs_index")

    
    
    # db = BiorxivDatabase()

    # # Fetch details for a given date interval
    # details = db.fetch_details('biorxiv', '2023-06-01/2023-06-01')
    # print(details.text)

    # # Fetch preprint publications for a given date interval
    # preprints = db.fetch_preprint_publications('biorxiv', '2023-06-01/2023-06-05')
    # print(preprints)

    # # Fetch published articles for a given date interval
    # articles = db.fetch_published_articles('2023-06-01/2023-06-05')
    # print(articles)
    
#  -------- DO NOT WORK BECAUSE OF THE API (below) -----------
    # Fetch summary statistics for a given date interval
    # stats = db.fetch_summary_statistics('2023-06-01/2023-06-05')
    # print(stats)

    # Fetch usage statistics for a given date interval
#     usage = db.fetch_usage_statistics('2023-06-01/2023-06-05')
#     print(usage)

#  -------- DO NOT WORK BECAUSE OF THE API (above) -----------

# #    ---------------- ENSEMBL API -----------------
    # ensembl = EnsemblDatabase()
    # sequence = ensembl.get_sequence_by_id("ENSG00000157764")
    # gene = ensembl.get_gene_by_id("ENSG00000157764")

    # print(sequence)
    # print(gene)


# ---------------- GEO API -----------------
    # db = "pubmed"  # example database
    # term = "breast cancer"  # example term
    # retmax = 1  # example maximum number of records to retrieve at once

    # geo = GeoDatabase(db)
    # for batch in geo.fetch_records(term, retmax):
    #     print(batch)
    

# ---------------- PUBMED API -----------------
    # pubmed = PubMed()

    # # Search for articles
    # search_results = pubmed.esearch("OpenAI")
    # print(search_results)

    # # Fetch specific articles
    # fetch_results = pubmed.efetch(['25359968', '26287646'])
    # print(fetch_results)

# ---------------- UNIPROT API -----------------
    # uniprot = UniProtDatabase()

    # protein0 = uniprot.search_proteins('P21802') -- NOT WORKING
    # protein1 = uniprot.get_protein_by_accession('P21802')
    # protein2 = uniprot.get_protein_isoforms_by_accession('P21802')
    # protein3 = uniprot.get_protein_sequence_by_accession('P21802')
    # protein4 = uniprot.get_protein_features_by_accession('P21802')
    # protein5 = uniprot.search_protein_features('insulin')
    # protein6 = uniprot.get_protein_variants_by_accession('P21802', 'isoform')
    # protein7 = uniprot.get_proteomics_by_accession('P21802')
    # protein8 = uniprot.get_antigen_by_accession('P21802')
    # protein9 = uniprot.get_mutagenesis_by_accession('P21802')


    # print(protein3)

if __name__ == '__main__':
    main()







import concurrent.futures
from typing import Dict, List
from pathlib import Path
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.node_parser import SimpleNodeParser
from llama_index.data_structs import Node
from llama_index.schema import Document
from llama_index import download_loader
from llama_index.schema import Document
from llama_index import GPTVectorStoreIndex
import ray
from ray.data import ActorPoolStrategy # to use Actors for parallelization -- map_batches() method

import os
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("Please add the OPENAI_API_KEY environment variable to run this script. Run the following in your terminal `export OPENAI_API_KEY=...`")


UnstructuredReader = download_loader("UnstructuredReader")
loader = UnstructuredReader()

def load_and_parse_json(json_row: Dict[str, str]) -> Dict[str, Document]:
    """
    Parse a row of JSON data into a Document object.

    Args:
        json_row (Dict[str, str]): A row of JSON data.

    Returns:
        Dict[str, Document]: A dictionary containing a "doc" key and a Document value.
    """
    # Fill the Document fields with data from the JSON.
    doc = Document(
        id=json_row.get('doi'),
        content=json_row.get('abstract'),
        authors=json_row.get('authors'),
        extra_info={"date": json_row.get('date'), "category": json_row.get('category')}
    )
    return {"doc": doc}

# def process_paper_data_to_document(paper_data: Dict) -> Document:
#     document = Document(
#         id=paper_data.get('doi'),  # or use another unique identifier if 'doi' is not suitable
#         title=paper_data.get('title'),
#         text=paper_data.get('abstract'),  # replace with the appropriate key
#         authors=paper_data.get('authors'),
#         date=paper_data.get('date'),
#         category=paper_data.get('category'),
#     )
#     return document


# # Step 1: Logic for parsing the files into llama_index documents - in our case it will be in formats of json (subject to change)
# def process_paper_data(paper_json):
#     '''
#     This function takes in a json object and returns a llama_index Document object
    
#     :param paper_json: a json object containing the data for a single paper
#     :return: a llama_index Document object
#     '''
#     paper_data = {}

#     paper_data['doi'] = paper_json['doi']
#     paper_data['title'] = paper_json['title']
#     paper_data['authors'] = paper_json['authors'].split('; ')
#     paper_data['date'] = paper_json['date']
#     paper_data['category'] = paper_json['category']
#     paper_data['abstract'] = paper_json['abstract']

#     # combining all relevant data from the papers into a single string
#     paper_text = f"Title: {paper_data['title']}\nAuthors: {', '.join(paper_data['authors'])}\nDate: {paper_data['date']}\nCategory: {paper_data['category']}\nAbstract: {paper_data['abstract']}"

#     # instantiate a Document object from llama_index.schema -- requires a doc_id and text and we will use the doi as the doc_id
#     document = Document(doc_id=paper_data['doi'], text=paper_text)

#     return document

def convert_documents_into_nodes(documents: Dict[str, Document]) -> Dict[str, Node]:
    '''
    This function takes in a dictionary of documents and returns a dictionary of nodes

    :param documents: a dictionary of documents
    :return: a dictionary of nodes
    '''
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents([document for document in documents])
    return [{"node": node} for node in nodes]


class EmbedNodes:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            # Use all-mpnet-base-v2 Sentence_transformer.
            # This is the default embedding model for LlamaIndex/Langchain.
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={"device": "cpu"},
            # Use GPU for embedding and specify a large enough batch size to maximize GPU utilization.
            # Remove the "device": "cuda" to use CPU instead.
            encode_kwargs={"device": "cpu", "batch_size": 100}
            )
    
    def __call__(self, node_batch: Dict[str, List[Node]]) -> Dict[str, List[Node]]:
        nodes = node_batch["node"]
        text = [node.text for node in nodes]
        embeddings = self.embedding_model.embed_documents(text)
        assert len(nodes) == len(embeddings)

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return {"embedded_nodes": nodes}



















# # Get the paths for the locally downloaded documentation.
# all_docs_gen = Path("../bio_papers").rglob("*")
# all_docs = [{"path": doc.resolve()} for doc in all_docs_gen]

# # Create the Ray Dataset pipeline
# ds = ray.data.from_items(all_docs)
# # Use `flat_map` since there is a 1:N relationship. Each filepath returns multiple documents.
# loaded_docs = ds.map_batches(load_and_parse_files)
# # Use `flat_map` since there is a 1:N relationship. Each document returns multiple nodes.
# nodes = loaded_docs.map_batches(convert_documents_into_nodes)
# # Use `map_batches` to specify a batch size to maximize GPU utilization.
# # We define `EmbedNodes` as a class instead of a function so we only initialize the embedding model once. 
# #   This state can be reused for multiple batches.
# embedded_nodes = nodes.map_batches(
#     EmbedNodes, 
#     batch_size=100, 
#     # Use 1 GPU per actor.
#     num_gpus=1,
#     # There are 4 GPUs in the cluster. Each actor uses 1 GPU. So we want 4 total actors.
#     # Set the size of the ActorPool to the number of GPUs in the cluster.
#     compute=ActorPoolStrategy(size=1), 
#     )


# # Step 5: Trigger execution and collect all the embedded nodes.
# bio_docs_nodes = []
# for row in embedded_nodes.iter_rows():
#     node = row["embedded_nodes"]
#     assert node.embedding is not None
#     bio_docs_nodes.append(node)

# # Step 6: Store the embedded nodes in a local vector store, and persist to disk.
# print("Storing Ray Documentation embeddings in vector index.")

# bio_docs_index = GPTVectorStoreIndex(nodes=bio_docs_nodes)
# bio_docs_index.storage_context.persist(persist_dir="/tmp/bio_docs_index")





























# Repeat the same steps for the Anyscale blogs
# Download the Anyscale blogs locally
# all_blogs_gen = Path("./www.anyscale.com/blog/").rglob("*")
# all_blogs = [{"path": blog.resolve()} for blog in all_blogs_gen]

# ds = ray.data.from_items(all_blogs)
# loaded_docs = ds.flat_map(load_and_parse_files)
# nodes = loaded_docs.flat_map(convert_documents_into_nodes)
# embedded_nodes = nodes.map_batches(
#     EmbedNodes, 
#     batch_size=100, 
#     compute=ActorPoolStrategy(size=4), 
#     num_gpus=1)





