from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import PubMedRetriever
from py2neo import Graph
import numpy as np
from sklearn.cluster import KMeans
from langchain.vectorstores import Chroma, FAISS
from sklearn.preprocessing import StandardScaler
from prompts import Coherent_sentences_template, Graph_Answer_Gen_Template, Graph_Answer_Gen_Template_airo, Graph_Answer_Gen_Template_alpaca
from langchain.prompts import PromptTemplate
from Custom_Agent import get_similar_compounds
from tqdm import tqdm
import streamlit as st
from sentence_transformers import SentenceTransformer

def get_node_label(graph: Graph, node_name: str) -> str:
    query = f"""
    MATCH (node)
    WHERE toLower(node.name) = toLower("{node_name}")
    RETURN head(labels(node)) AS FirstLabel
    """
    result = graph.run(query).data()
    if result:
        print(query)
        return result[0]['FirstLabel']
    else:
        return None

def get_source_and_target_paths(graph: Graph, names: List[str]) -> Tuple[List[Relationship], List[Relationship]]:
    source_label = get_node_label(graph, names[0])
    target_label = get_node_label(graph, names[1])

    query_source = f"""
    MATCH path=(source:{source_label})-[*1..2]->(node)
    WHERE toLower(source.name) = toLower("{names[0]}")
    WITH relationships(path) AS rels, nodes(path) AS nodes
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """

    query_target = f"""
    MATCH path=(target:{target_label})-[*1..2]->(node)
    WHERE toLower(target.name) = toLower("{names[1]}")
    WITH relationships(path) AS rels, nodes(path) AS nodes
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """
    print(query_source)
    print(query_target)
    source_results = list(graph.run(query_source))
    target_results = list(graph.run(query_target))
    source_paths = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in source_results]
    target_paths = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in target_results]
    print("source paths:")
    print(len(source_paths))
    print("target paths")
    print(len(target_paths))
    source_relationships = [construct_relationship_string(record['path_nodes'], record['path_relationships']) for record in source_results]
    target_relationships = [construct_relationship_string(record['path_nodes'], record['path_relationships']) for record in target_results]
    
    with open("sample.txt", "w") as file:
        source_rels_to_write = [construct_path_string(record['path_nodes'], record['path_relationships']) for record in source_results]
        for string in source_rels_to_write:
            file.write(string + '\n')
        target_rels_to_write = [construct_path_string(record['path_nodes'], record['path_relationships']) for record in target_results]
        for string in target_rels_to_write:
            file.write(string + '\n')
    
    return source_paths, target_paths, source_relationships, target_relationships

def construct_path_string(nodes: List[str], relationships: List[str]) -> str:
    path_elements = []
    for node, relationship in zip(nodes, relationships):
        if node is None or relationship is None:
            continue  # Skip this element if the node or the relationship is None
        path_elements.append(f"{node} -> {relationship}")
    if nodes[-1] is not None:
        path_elements.append(nodes[-1])  # add the last node
    return " -> ".join(path_elements)

def construct_relationship_string(nodes: List[str], relationships: List[str]) -> str:
    path_elements = []
    for i in range(len(nodes) - 1):
        if nodes[i] is None or relationships[i] is None or nodes[i + 1] is None:
            continue  # Skip this element if any of the nodes or the relationship is None
        path_elements.append(f"{nodes[i]} -> {relationships[i]} -> {nodes[i + 1]}")
    return ", ".join(path_elements)

def find_shortest_paths(graph: Graph, names: List[str], entity_types: Dict[str, str], repeat: bool) -> List[Dict[str, Any]]:
    source_label = get_node_label(graph, names[0])
    target_label = get_node_label(graph, names[1])
    names_conditions = f'WHERE toLower(source.name) = toLower("{names[0]}") AND toLower(target.name) = toLower("{names[1]}")'
    query = f"""
    MATCH (source:{source_label }), (target:{target_label})
    {names_conditions}
    MATCH p = allShortestPaths((source)-[*]-(target))
    WITH p, [rel IN relationships(p) | type(rel)] AS path_relationships
    WITH relationships(p) AS rels, nodes(p) AS nodes, path_relationships
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """
    print(query)
    result = graph.run(query)

    if not result:
        source_entity_type = entity_types[f"{names[0]}"]
        target_entity_type = entity_types[f"{names[1]}"]
        if source_entity_type == 'Drug':
            source_test_query = f"""
            MATCH (p:Drug)
            WHERE p.name = "{names[0]}"
            RETURN p
            """
            source_test_result = graph.run(source_test_query)
            if not source_test_result:
                similar_compounds = get_similar_compounds({names[0]}, 20)
                for compound in similar_compounds[1:]:
                    source_similar_compounds_test_query = f"""
                    MATCH (p:Drug)
                    WHERE p.name = "{compound}"
                    RETURN p
                    """
                    print(source_similar_compounds_test_query)           # start from index 1 to skip the first similar compound
                    source_similar_compounds_test_result = graph.run(source_similar_compounds_test_query)
                    if source_similar_compounds_test_result:
                        names[0] = compound
                        break

        if target_entity_type == 'Drug':
            target_test_query = f"""
            MATCH (p:Drug)
            WHERE p.name = "{names[1]}"
            RETURN p
            """            
            target_test_result = graph.run(target_test_query)
            if not target_test_result:
                similar_compounds = get_similar_compounds({names[1]}, 20)
                for compound in similar_compounds[1:]:  # start from index 1 to skip the first similar compound
                    target_similar_compounds_test_query = f"""
                    MATCH (p:Drug)
                    WHERE p.name = "{compound}"
                    RETURN p
                    """
                    print(target_similar_compounds_test_query)             # start from index 1 to skip the first similar compound
                    target_similar_compounds_test_result = graph.run(target_similar_compounds_test_query)
                    if target_similar_compounds_test_result:
                        names[1] = compound
                        break

    result = graph.run(query)
    #if not result and source_entity_type == "Drug":
    # Initialize a set to store unique associated genes
    unique_source_paths = []
    unique_target_paths = []
    unique_graph_rels = set()

    unique_rel_paths = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in result]

    for record in result:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        rel_string = construct_relationship_string(path_nodes, path_relationships)
        unique_graph_rels.add(rel_string)
    
    source_paths, target_paths, source_relationships, target_relationships = get_source_and_target_paths(graph, names)

    for path in source_paths:
        unique_source_paths.append(path)
    # Construct and add the target path relationship strings to the list
    for path in target_paths:
        unique_target_paths.append(path)

    for rel in source_relationships:
        unique_graph_rels.add(rel)

    for rel in target_relationships:
        unique_graph_rels.add(rel)

    # Convert unique_relationships set to list
    unique_source_paths_list = list(unique_source_paths)
    unique_target_paths_list = list(unique_target_paths)
    unique_graph_rels_list = list(unique_graph_rels)

    return unique_rel_paths, unique_target_paths_list, unique_source_paths_list, unique_graph_rels_list

def query_inter_relationships_direct1(graph: Graph, node:str) -> Tuple[List[Dict[str, Any]], List[str], List[str], Set[str], List[str]]:
    node_label = get_node_label(graph, node)
    all_nodes = set()
    graph_strings = set()
    relationships_direct = set()
    og_relationships_direct_list = []
    direct_nodes = []

    direct_relationships_query = f"""
    MATCH path=(n:{node_label})-[r]-(m)
    WHERE n.name = "{node}" AND n.name IS NOT NULL
    WITH nodes(path) AS nodes, relationships(path) AS rels
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """
    result_direct = list(graph.run(direct_relationships_query, node=node))
    direct_nodes.extend([node for record in result_direct for node in record['path_nodes']])
    print(direct_relationships_query)

    for record in result_direct:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        rel_string = construct_path_string(path_nodes, path_relationships)
        graph_string = construct_relationship_string(path_nodes, path_relationships)
        graph_strings.add(graph_string)
        relationships_direct.add(rel_string)
        all_nodes.update(record['path_nodes'])
        og_relationships_direct_list.append({'nodes': path_nodes, 'relationships': path_relationships})

    graph_strings_list = list(graph_strings)
    og_relationships_direct_list = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in result_direct]
    return og_relationships_direct_list, graph_strings_list, all_nodes, direct_nodes

def query_inter_relationships_between_direct(graph: Graph, direct_nodes, nodes:List[str]) -> str:
    node_labels = [get_node_label(graph, node) for node in nodes + direct_nodes]
    unique_labels = list(set(node_labels))
    
    query_parameters_2 = {"nodes": list(nodes) + direct_nodes, "unique_labels": unique_labels}
    total_nodes = list(nodes) + direct_nodes
    print("number of direct nodes")
    print(len(total_nodes))
    # Query for paths between the nodes from the original list

    inter_between_direct_query = """
    MATCH (n)
    WHERE n.name IN $nodes AND any(label in labels(n) WHERE label IN $unique_labels)
    CALL apoc.path.spanningTree(n, {minLevel: 1, maxLevel: 3, limit: 100}) YIELD path
    WITH nodes(path) AS nodes, relationships(path) AS rels
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """

    result_inter_direct = list(graph.run(inter_between_direct_query, **query_parameters_2))
    print(inter_between_direct_query)

    all_nodes = set()
    graph_strings = set()

    for record in result_inter_direct:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        graph_string = construct_relationship_string(path_nodes, path_relationships)
        graph_strings.add(graph_string)
        all_nodes.update(record['path_nodes'])
    relationships_inter_direct_list = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in result_inter_direct]
    graph_strings_list = list(graph_strings)
    print("number of inter direct relations:")
    print(len(relationships_inter_direct_list))
    return relationships_inter_direct_list, graph_strings_list, all_nodes

#######################################################################################################################################################################################

def generate_answer(llm, relationships_list, source_list, target_list, inter_direct_list, inter_direct_inter, question, source, target, gene_string: Optional[str] = None):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template, input_variables=["input", "question"])
    #prompt = PromptTemplate(template=Graph_Answer_Gen_Template_alpaca, input_variables=["input", "question"])
    gen_chain = LLMChain(llm=llm, prompt=prompt)
    multi_hop = ', '.join(relationships_list)
    source_sentences = ','.join(source_list)
    target_sentences = ','.join(target_list)
    Inter_relationships = inter_direct_list + inter_direct_inter
    Inter_sentences = ','.join(Inter_relationships)
    sep_1 = f"Indirect relations between {source} and {target}:"
    sep2 = f"Direct relations from {source}:"
    sep3 = f"Direct relations from {target}:"
    sep4 = f"Relations between the targets of {source} and {target}"
    if gene_string:
        sentences = '\n'.join([sep_1, multi_hop, sep2, source_sentences, sep3, target_sentences, sep4, Inter_sentences, gene_string])
    else:
        sentences = '\n'.join([sep_1, multi_hop, sep2, source_sentences, sep3, target_sentences, sep4, Inter_sentences])
    answer = gen_chain.run(input=sentences, question=question)
    print(answer)
    return answer

def generate_answer_airo(llm, relationships_list, source_list, target_list, inter_direct_list, inter_direct_inter, question, source, target, gene_string: Optional[str] = None):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template_airo, input_variables=["question", 
                                                                                      "source", 
                                                                                      "target", 
                                                                                      "multihop_relations", 
                                                                                      "direct_relations_source",
                                                                                      "direct_relations_target",
                                                                                      "inter_relations"])
    gen_chain = LLMChain(llm=llm, prompt=prompt)
    multi_hop = ', '.join(relationships_list)
    source_sentences = ','.join(source_list)
    target_sentences = ','.join(target_list)
    Inter_relationships = inter_direct_list + inter_direct_inter
    Inter_sentences = ','.join(Inter_relationships)

    answer = gen_chain.run(question=question,
                           source=source,
                           target=target,
                           multihop_relations=multi_hop,
                           direct_relations_source=source_sentences,
                           direct_relations_target=target_sentences,
                           inter_relations=Inter_sentences)
    print(answer)
    return answer

#########################################################################################################################################################################################

def cluster_and_select_med(paths_list, n_cluster, progress_callback=None):
    model = SentenceTransformer('pritamdeka/S-Bluebert-snli-multinli-stsb')
    #pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT
    #pritamdeka/S-Bluebert-snli-multinli-stsb
    #pritamdeka/S-PubMedBert-MS-MARCO
    #pritamdeka/S-Bluebert-snli-multinli-stsb
    #print(paths_list)
    sentences_list = [construct_path_string(path['nodes'], path['relationships']) for path in paths_list]
    batch_size = 2048
    total_iterations = len(sentences_list) // batch_size + 1

    embeddings_list = []
    for i in range(0, len(sentences_list), batch_size):
        batch_sentences = sentences_list[i:i+batch_size]

        # Embed documents for the batch
        batch_embeddings_array = np.array(model.encode(batch_sentences, convert_to_tensor=True).cpu())
        embeddings_list.append(batch_embeddings_array)

        # Update the progress bar
        if progress_callback:
            progress_callback((i + len(batch_sentences)) / len(sentences_list))

    # Concatenate embeddings from all batches
    embeddings_array = np.concatenate(embeddings_list)

    # Continue with the remaining code
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(embeddings_array)

    n_clusters = n_cluster
    kmeans = KMeans(n_clusters=n_clusters, init="random", n_init=10, max_iter=300, random_state=42)
    kmeans.fit(scaled_features)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    cluster_documents = {}
    for i, label in enumerate(cluster_labels):
        document = sentences_list[i]
        if label not in cluster_documents:
            cluster_documents[label] = document

    final_result = list(cluster_documents.values())
    print("done clustering")
    return final_result

def embed_and_select_med(paths_list, question, n_embed):
    sentences_list = [construct_path_string(path['nodes'], path['relationships']) for path in paths_list]
    hf = HuggingFaceEmbeddings(
    model_name='pritamdeka/S-Bluebert-snli-multinli-stsb',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True})

    db = Chroma.from_texts(sentences_list, hf)
    retriever = db.as_retriever(search_kwargs={"k": n_embed})
    docs = retriever.get_relevant_documents(question)[:n_embed]

    final_result = [doc.page_content for doc in docs]
    del db, retriever, docs, hf, sentences_list
    print("done embedding")
    return final_result

def select_paths(paths, question, n_cluster, n_embed, progress_callback):
    clustered_paths = cluster_and_select_med(paths, n_cluster, progress_callback)
    selected_paths_stage1 = [path for path in paths if construct_path_string(path['nodes'], path['relationships']) in clustered_paths and None not in path['nodes']]

    # Create a dictionary mapping string representations to original paths
    path_dict = {construct_path_string(path['nodes'], path['relationships']): path for path in selected_paths_stage1}

    embedded_paths = embed_and_select_med(selected_paths_stage1, question, n_embed)
    selected_paths_stage2 = [path_dict[path_str] for path_str in embedded_paths]
    
    selected_nodes = [node for path in selected_paths_stage2 for node in path['nodes']]
    paths_list = [construct_path_string(path['nodes'], path['relationships']) for path in selected_paths_stage2]
    paths_list = list(set(paths_list))
    unique_rels_list = [construct_relationship_string(path['nodes'], path['relationships']) for path in selected_paths_stage2]
    unique_rels_list = list(set(unique_rels_list))
    return paths_list, selected_nodes, unique_rels_list, selected_paths_stage2

def select_paths2(paths, question, n_cluster, n_embed, progress_callback):
    clustered_paths = cluster_and_select_med(paths, n_cluster, progress_callback)
    selected_paths_stage1 = [path for path in paths if construct_path_string(path['nodes'], path['relationships']) in clustered_paths and None not in path['nodes']]

    # Create a dictionary mapping string representations to original paths
    path_dict = {construct_path_string(path['nodes'], path['relationships']): path for path in selected_paths_stage1}

    embedded_paths = embed_and_select_med(selected_paths_stage1, question, n_embed)
    selected_paths_stage2 = [path_dict[path_str] for path_str in embedded_paths]
    
    selected_nodes = [node for path in selected_paths_stage2 for node in path['nodes']]
    paths_list = [construct_path_string(path['nodes'], path['relationships']) for path in selected_paths_stage2]
    paths_list = list(set(paths_list))
    unique_rels_list = [construct_relationship_string(path['nodes'], path['relationships']) for path in selected_paths_stage2]
    unique_rels_list = list(set(unique_rels_list))
    return paths_list, selected_nodes, unique_rels_list

#####################################################################################################################################################################################################

class KnowledgeGraphRetrieval:
    # 5, 5, 3, 3, 10, 30, 10
    def __init__(self, uri, username, password, llm, entity_types):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entity_types = entity_types  # Store the entity_types dictionary as an instance variable

    def _call(self, names_list, question, generate_an_answer, related_interactions, progress_callback=None):
        if related_interactions == True:
            result = find_shortest_paths(self.graph, names_list, self.entity_types, repeat=True)
        else:
            result = find_shortest_paths(self.graph, names_list, self.entity_types, repeat=True)

        # Check if result is a tuple of length 2
        if isinstance(result, tuple) and len(result) == 6:
            # Unpack result into relationship_context and associated_genes_string
            unique_relationships_list, unique_target_paths_list, unique_source_paths_list, unique_graph_rels = result
        else:
            # If not, relationship_context is result and associated_genes_string is an empty string
            unique_relationships_list, unique_target_paths_list, unique_source_paths_list, unique_graph_rels  = result
            #gene_string = ""

        final_target_paths, selected_target_nodes, target_unique_rels, selected_target_paths = select_paths(unique_target_paths_list, question, len(unique_target_paths_list)//15, 3, progress_callback)
        print("final_target_paths")
        print(len(final_target_paths))

        final_source_paths, selected_source_nodes, source_unique_rels, selected_source_paths = select_paths(unique_source_paths_list, question, len(unique_source_paths_list)//15, 3, progress_callback)
        print("final_source_paths")
        print(len(final_source_paths))
        
        final_inter_relationships, selected_inter_nodes, inter_unique_rels, selected_inter_paths = select_paths(unique_relationships_list, question, len(unique_relationships_list), len(unique_relationships_list), progress_callback)
        print("final_inter_relationships")
        print(len(final_inter_relationships))
        
        query_nodes = selected_target_nodes + selected_source_nodes + selected_inter_nodes
        query_nodes = (set(query_nodes))
        names_set = set(names_list)
        #query_nodes.update(final_path_nodes)
        all_first_nodes = list(query_nodes)
        query_nodes = [name for name in query_nodes if name.lower() not in names_set]
        print("query nodes")
        print(len(query_nodes))
        print(query_nodes)

        og_target_direct_relations = set()
        selected_inter_direct_nodes = set()
        inter_direct_unique_graph_rels = set()
        final_selected_target_direct_paths = []

        for node in query_nodes:
            target_direct_relations, inter_direct_graph_rels, source_and_target_nodes1, direct_nodes = query_inter_relationships_direct1(self.graph, node)
            if target_direct_relations:
                if len(target_direct_relations) < 15:
                    inter_direct_relationships, selected_nodes, inter_direct_unique_rels, selected_target_direct_paths = select_paths(target_direct_relations, question, len(target_direct_relations), 3, progress_callback)
                else:
                    inter_direct_relationships, selected_nodes, inter_direct_unique_rels, selected_target_direct_paths = select_paths(target_direct_relations, question, len(target_direct_relations)//15, 3, progress_callback)
                og_target_direct_relations.update(inter_direct_relationships)
                selected_inter_direct_nodes.update(selected_nodes)
                inter_direct_unique_graph_rels.update(inter_direct_unique_rels)
                final_selected_target_direct_paths.append(selected_target_direct_paths)
                print("success")
                print(len(inter_direct_relationships))
                print(inter_direct_relationships)
            else:
                print("skipping")
                continue
        print("nodes before clustering and embedding")
        print(len(selected_inter_direct_nodes))
        #flat_selected_target_direct_paths = [item for sublist in final_selected_target_direct_paths for item in sublist]
        #final_inter_direct_relationships, final_selected_nodes, final_inter_direct_rels, selected_inter_direct_paths = select_paths(paths=flat_selected_target_direct_paths, question=question, n_cluster=len(list(og_target_direct_relations))//15, n_embed=30, progress_callback=progress_callback)
        
        final_inter_direct_relationships = list(og_target_direct_relations)
        #final_selected_inter_direct_nodes = list(set(final_selected_nodes))
        #final_inter_direct_unique_graph_rels = list(set(final_inter_direct_rels))
        final_selected_inter_direct_nodes = list(set(selected_inter_direct_nodes))
        final_inter_direct_unique_graph_rels = list(set(inter_direct_unique_graph_rels))
        #og_target_direct_relations, relationships_inter_direct_list, inter_direct_unique_graph_rels, source_and_target_nodes1, direct_nodes = query_inter_relationships_direct1(self.graph, query_nodes)
        print("number of unique inter_direct_relationships:")
        print(len(final_inter_direct_relationships))

        if final_inter_direct_relationships:
            
            target_inter_relations, inter_direct_inter_unique_graph_rels, source_and_target_nodes2 = query_inter_relationships_between_direct(self.graph, final_selected_inter_direct_nodes, query_nodes)
            if len(target_inter_relations) < 50:
                final_inter_direct_inter_relationships, selected_inter_direct_inter_nodes, inter_direct_inter_unique_rels = select_paths2(target_inter_relations, question, len(target_inter_relations), 10, progress_callback)
            else:
                final_inter_direct_inter_relationships, selected_inter_direct_inter_nodes, inter_direct_inter_unique_rels = select_paths2(target_inter_relations, question, len(target_inter_relations)//15, 5, progress_callback)

        else:
            final_inter_direct_relationships = []
            selected_inter_direct_nodes = []

            target_inter_relations, inter_direct_inter_unique_graph_rels, source_and_target_nodes2 = query_inter_relationships_between_direct(self.graph, query_nodes, query_nodes)
            if target_inter_relations:
                final_inter_direct_inter_relationships, selected_inter_direct_inter_nodes, inter_direct_inter_unique_rels = select_paths2(target_inter_relations, question, len(target_inter_relations), 10, progress_callback)
            else:
                final_inter_direct_inter_relationships = []
                selected_inter_direct_inter_nodes = []

        print("final_inter_direct_inter_relationships")
        print(len(final_inter_direct_inter_relationships))
        all_nodes = set()
        if selected_inter_direct_nodes:
            all_nodes.update(selected_inter_direct_nodes)
        if selected_inter_direct_inter_nodes:
            all_nodes.update(selected_inter_direct_inter_nodes)
        all_nodes.update(query_nodes)
        print("all nodes:")
        print(len(all_nodes))
        #print(all_nodes)

        all_unique_graph_rels = set()
        all_unique_graph_rels.update(target_unique_rels)
        all_unique_graph_rels.update(source_unique_rels)
        all_unique_graph_rels.update(inter_unique_rels)
        all_unique_graph_rels.update(final_inter_direct_unique_graph_rels)
        all_unique_graph_rels.update(inter_direct_inter_unique_rels)


########################################################################################################
        
        if generate_an_answer == True:
            #final_context = generate_answer_airo(llm=self.llm,
            final_context = generate_answer(llm=self.llm, 
                                            relationships_list=final_inter_relationships,
                                            question=question,
                                            source_list=final_source_paths,
                                            target_list=final_target_paths,
                                            inter_direct_list=final_inter_direct_relationships,
                                            inter_direct_inter=final_inter_direct_inter_relationships,
                                            source=names_list[0],
                                            target=names_list[1]
                                            )
                                            

            answer = final_context

        response = {"result": answer, 
                    "multi_hop_relationships": final_inter_relationships,
                    "source_relationships": final_source_paths,
                    "target_relationships": final_target_paths,
                    "inter_direct_relationships": final_inter_direct_relationships,
                    "inter_direct_inter_relationships": final_inter_direct_inter_relationships,
                    "all_nodes": all_nodes,
                    "all_rels": all_unique_graph_rels}
        return response





