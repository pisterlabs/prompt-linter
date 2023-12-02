from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.chains.llm import LLMChain
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from py2neo import Graph
import numpy as np
from langchain.prompts import PromptTemplate
import gc

from CustomLibrary.Graph_Queries import (
    query_direct, 
    query_between_direct,
    get_node_label,
    get_node_labels_dict
    )

from CustomLibrary.Graph_Utils import (
    select_paths,
    select_paths_pharos
)

from CustomLibrary.Custom_Prompts import Graph_Answer_Gen_Template_alpaca

from CustomLibrary.Pharos_Queries import (
    ligand_query,
    target_query,
    disease_query
)

def generate_answer(llm, entities_list, question, start_paths, mid_paths, inter_direct_inter):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template_alpaca, input_variables=["input", "question"])
    #prompt = PromptTemplate(template=Graph_Answer_Gen_Template_alpaca, input_variables=["input", "question"])
    gen_chain = LLMChain(llm=llm, prompt=prompt)
    start_paths = ','.join(start_paths)
    Inter_relationships = mid_paths + inter_direct_inter
    Inter_sentences = ','.join(Inter_relationships)
    sep1 = f"Starting paths from {entities_list}:"
    sep2 = f"Intermediate paths of {entities_list}:"
    sentences = '\n'.join([sep1, start_paths, sep2, Inter_sentences])
    answer = gen_chain.run(input=sentences, question=question)
    print(answer)
    return answer

class PharosGraphQA:
    def __init__(self, uri, username, password, llm, entities_list, constituents_dict, constituents_paths):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entities_list = entities_list
        self.constituents_dict = constituents_dict
        self.constituents_paths = constituents_paths
        
    def _call(self, question, progress_callback=None):
        start_paths = []
        start_nodes = []
        start_graph_rels = []

        for entity in self.entities_list:
            entity_name, entity_type = entity
            if entity_type == "Disease":
                result = disease_query(entity_name, question)
            elif entity_type == "Drug" or "Food" or "Metabolite":
                result = ligand_query(entity_name, question)
            elif entity_type == "Gene":
                result = target_query(entity_name, question)

            if result is not None:
                (Pharos_Paths,
                Pharos_Nodes, 
                Pharos_Rels) = result
                start_paths.extend(Pharos_Paths)
                start_nodes.extend(Pharos_Nodes)
                start_graph_rels.extend(Pharos_Rels)

            if entity in self.constituents_dict and self.constituents_dict[entity]:
                constituents = self.constituents_dict[entity]
                constituents = [constituent for constituent in constituents if constituent != 'None']
                if constituents and 'None' not in constituents:
                    for constituent in constituents:
                        constituent_name, constituent_type = constituent
                        if constituent_type == "Disease":
                            result = disease_query(constituent_name, question)
                        elif constituent_type == "Drug" or "Food" or "Metabolite":
                            result = ligand_query(constituent_name, question)
                        elif constituent_type == "Gene":
                            result = target_query(constituent_name, question)

                        if result is not None:
                            (Constituent_Pharos_Paths,
                            Constituent_Pharos_Nodes, 
                            Constituent_Pharos_Rels) = result

                            start_paths.extend(Constituent_Pharos_Paths)
                            start_nodes.extend(Constituent_Pharos_Nodes)
                            start_graph_rels.extend(Constituent_Pharos_Rels)
                else:
                    continue
            
        print("start_paths:", len(start_paths))
        print("start_nodes:", len(start_nodes))
        print("start_graph_rels:", len(start_graph_rels))

        query_nodes = list(set(start_nodes))

        print("query nodes")
        print(len(query_nodes))
        print(query_nodes)

        mid_direct_paths = set()
        mid_direct_nodes = set()
        mid_direct_graph_rels = set()
        
        node_labels = get_node_labels_dict(self.graph, query_nodes)
        for node in query_nodes:
            node_label = node_labels.get(node)
            if node_label is not None:
                paths = query_direct(self.graph, node, node_label)
                if paths:
                    (selected_paths, 
                    selected_nodes, 
                    selected_graph_rels) = select_paths(paths, 
                                                        question, 
                                                        len(paths)//3, 
                                                        3,
                                                        progress_callback)
                    
                    mid_direct_paths.update(selected_paths)
                    mid_direct_nodes.update(selected_nodes)
                    mid_direct_graph_rels.update(selected_graph_rels)
                    print("success")
                    print(len(selected_paths))
                    print(selected_paths)
                    del paths, selected_paths, selected_nodes, selected_graph_rels
                    gc.collect()
                else:
                    print("skipping")
                    continue

        print("number of unique inter_direct_relationships:")
        print(len(mid_direct_paths))

        mid_inter_paths = query_between_direct(self.graph, 
                                                list(mid_direct_nodes), 
                                                query_nodes)
        

        n_cluster = max(len(mid_inter_paths)//10, 1)
        if n_cluster > 30:
            n_embed = 30
        else:
            n_embed = n_cluster

        print("n_cluster:", n_cluster)
        (selected_mid_inter_paths, 
         selected_mid_inter_nodes, 
         selected_mid_inter_graph_rels) = select_paths(mid_inter_paths, 
                                                        question, 
                                                        n_cluster, 
                                                        n_embed,
                                                        progress_callback)
        
        print("final_inter_direct_inter_relationships")
        print(len(selected_mid_inter_paths))

        all_graph_rels = set()
        all_graph_rels.update(selected_mid_inter_graph_rels)
        all_graph_rels.update(mid_direct_graph_rels)
        all_graph_rels.update(start_graph_rels)

        all_graph_rels = list(all_graph_rels)
        print("all_graph_rels")
        print(len(all_graph_rels))

########################################################################################################
        
        params = {
            "llm": self.llm,
            "entities_list": self.entities_list,
            "question": question,
            "start_paths": start_paths,
            "mid_paths": list(mid_direct_paths),
            "inter_direct_inter": list(selected_mid_inter_paths),
        }

        final_context = generate_answer(**params)

        answer = final_context

        response = {"result": answer,
                    "all_rels": all_graph_rels}
                
        return response