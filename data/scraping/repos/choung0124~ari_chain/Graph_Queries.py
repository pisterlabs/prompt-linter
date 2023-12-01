from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from py2neo import Graph
import numpy as np
from sklearn.cluster import KMeans
from langchain.vectorstores import Chroma, FAISS
from sklearn.preprocessing import StandardScaler
from prompts import Coherent_sentences_template, Graph_Answer_Gen_Template
from langchain.prompts import PromptTemplate
from Custom_Agent import get_similar_compounds

def get_source_and_target_paths(graph: Graph, label: str, names: List[str]) -> Tuple[List[Relationship], List[Relationship]]:
    query_source = f"""
    MATCH path=(source:{label})-[*1..2]->(node)
    WHERE toLower(source.name) = toLower("{names[0]}")
    WITH relationships(path) AS rels, nodes(path) AS nodes
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """
    query_target = f"""
    MATCH path=(target:{label})-[*1..2]->(node)
    WHERE toLower(target.name) = toLower("{names[1]}")
    WITH relationships(path) AS rels, nodes(path) AS nodes
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """
    print(query_source)
    print(query_target)
    source_results = list(graph.run(query_source))
    target_results = list(graph.run(query_target))
    source_paths = [construct_path_string(record['path_nodes'], record['path_relationships']) for record in source_results]
    target_paths = [construct_path_string(record['path_nodes'], record['path_relationships']) for record in target_results]
    all_path_nodes = set()
    for record in source_results:
        all_path_nodes.update(record['path_nodes'])
    for record in target_results:
        all_path_nodes.update(record['path_nodes'])
    print("source and target nodes:")
    print(len(all_path_nodes))
    print(all_path_nodes)
    
    return source_paths, target_paths, all_path_nodes

def construct_path_string(nodes: List[str], relationships: List[str]) -> str:
    path_elements = [f"{nodes[i]} -> {relationships[i]} -> {nodes[i + 1]}" for i in range(len(nodes) - 1)]
    return " -> ".join(path_elements)

def find_shortest_paths(graph: Graph, label: str, names: List[str], entity_types: Dict[str, str], repeat: bool) -> List[Dict[str, Any]]:

    names_conditions = f'WHERE toLower(source.name) = toLower("{names[0]}") AND toLower(target.name) = toLower("{names[1]}")'
    query = f"""
    MATCH (source:{label}), (target:{label})
    {names_conditions}
    MATCH p = allShortestPaths((source)-[*]-(target))

    WITH p, [r IN relationships(p) WHERE type(r) = "ASSOCIATED_WITH" | startNode(r).name] AS associated_genes
    WITH p, associated_genes, [rel IN relationships(p) | type(rel)] AS path_relationships

    RETURN [node IN nodes(p) | node.name] AS path_nodes, associated_genes, path_relationships
    """
    print(query)
    result = graph.run(query)
    if not result and repeat==True:
        source_entity_type = entity_types[f"{names[0]}"]
        target_entity_type = entity_types[f"{names[1]}"]
        if source_entity_type == 'Drug':
            source_test_query = f"""
            MATCH (p:{label})
            WHERE p.name = "{names[0]}"
            RETURN p
            """
            source_test_result = graph.run(source_test_query)
            if not source_test_result:
                similar_compounds = get_similar_compounds({names[0]}, 20)
                for compound in similar_compounds[1:]:
                    source_similar_compounds_test_query = f"""
                    MATCH (p:{label})
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
            MATCH (p:{label})
            WHERE p.name = "{names[1]}"
            RETURN p
            """            
            target_test_result = graph.run(target_test_query)
            if not target_test_result:
                similar_compounds = get_similar_compounds({names[1]}, 20)
                for compound in similar_compounds[1:]:  # start from index 1 to skip the first similar compound
                    target_similar_compounds_test_query = f"""
                    MATCH (p:{label})
                    WHERE p.name = "{compound}"
                    RETURN p
                    """
                    print(target_similar_compounds_test_query)             # start from index 1 to skip the first similar compound
                    target_similar_compounds_test_result = graph.run(target_similar_compounds_test_query)
                    if target_similar_compounds_test_result:
                        names[1] = compound
                        break

    result = graph.run(query)

    associated_genes_set = set()
    unique_relationships = set()
    unique_source_paths = set()
    unique_target_paths = set()
    final_path_nodes = set()

    for record in result:
        path_nodes = record['path_nodes']
        associated_genes_list = record['associated_genes']
        path_relationships = record['path_relationships']
        # Add the genes to the set
        final_path_nodes.update(record['path_nodes'])
        # Construct and add the relationship strings to the set
        rel_string = construct_path_string(path_nodes, path_relationships)
        unique_relationships.add(rel_string)
    
    source_paths, target_paths, source_and_target_nodes = get_source_and_target_paths(graph, label, names)

    for path in source_paths:
        unique_source_paths.add(path)

    # Construct and add the target path relationship strings to the set
    for path in target_paths:
        unique_target_paths.add(path)

    for node in source_and_target_nodes:
        final_path_nodes.add(node)

    print("number of nodes:")
    print(len(final_path_nodes))
    print(final_path_nodes)
    # Remove the source and target node names from the associated genes set
    lower_names = {name.lower() for name in names}
    associated_genes_set = {gene for gene in associated_genes_set if gene.lower() not in lower_names}

    # Convert unique_relationships set to list
    unique_relationships_list = list(unique_relationships)
    unique_source_paths_list = list(unique_source_paths)
    unique_target_paths_list = list(unique_target_paths)

    # Check if there are associated genes and return accordingly
    if associated_genes_set:
        associated_genes_list = list(associated_genes_set)
        gene_string = f"The following genes are associated with both {names[0]} and {names[1]}: {', '.join(associated_genes_list)}"
        print(gene_string)
        return unique_relationships_list, unique_target_paths_list, unique_source_paths_list, final_path_nodes, gene_string
    else:
        print("There are no associated genes.")
        #print(unique_relationships_list)

        return unique_relationships_list, unique_target_paths_list, unique_source_paths_list, final_path_nodes
    
def query_inter_relationships(graph: Graph, nodes:List[str]) -> str:
    nodes_string = "[" + ", ".join("$node" + str(i) for i in range(len(nodes))) + "]"
    query_parameters = {"nodes": list(nodes)}

    # Query for interrelationships among nodes
    inter_relationships_query = """
    MATCH (n:Entity) WHERE n.name IN $nodes
    WITH collect(n) as nodes
    UNWIND nodes as n
    UNWIND nodes as m
    WITH * WHERE id(n) < id(m)
    MATCH path = allShortestPaths((n)-[*]-(m))
    WITH nodes(path) AS nodes, relationships(path) AS rels
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """

    result_inter = graph.run(inter_relationships_query, **query_parameters)
    print(inter_relationships_query)

    # Query for direct relationships to and from nodes
    direct_relationships_query = """
    MATCH p = (n)-[r]-(m)
    WHERE n.name IN $nodes
    WITH p, [rel IN relationships(p) | type(rel)] AS path_relationships
    RETURN [node IN nodes(p) | node.name] AS path_nodes, path_relationships
    """

    result_direct = graph.run(direct_relationships_query, **query_parameters)
    print(direct_relationships_query)
    
    # Combine results
    relationships_inter = set()
    relationships_direct = set()
    all_nodes = set()
    for record in result_inter:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        rel_string = construct_path_string(path_nodes, path_relationships)
        relationships_inter.add(rel_string)
        all_nodes.update(record['path_nodes'])
    for record in result_direct:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        rel_string = construct_path_string(path_nodes, path_relationships)
        relationships_direct.add(rel_string)
        all_nodes.update(record['path_nodes'])

    relationships_inter_list = list(relationships_inter) if relationships_inter else []
    relationships_direct_list = list(relationships_direct) if relationships_direct else []
    return relationships_inter_list, relationships_direct_list, all_nodes