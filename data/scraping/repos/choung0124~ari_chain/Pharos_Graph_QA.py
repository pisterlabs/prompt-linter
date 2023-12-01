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
    select_paths2, 
)

from CustomLibrary.Custom_Prompts import Graph_Answer_Gen_Template2, Graph_Answer_Gen_Template2_alpaca

from CustomLibrary.Pharos_Queries import (
    ligand_query,
    target_query,
    disease_query
)

def generate_answer(llm, source_list, target_list, inter_direct_list, inter_direct_inter, question, previous_answer, source, target, additional_list:Optional[List[str]]=None):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template2_alpaca, input_variables=["input", "question", "previous_answer"])
    #prompt = PromptTemplate(template=Graph_Answer_Gen_Template_alpaca, input_variables=["input", "question"])
    gen_chain = LLMChain(llm=llm, prompt=prompt)
    source_rels = ', '.join(source_list)
    target_rels = ','.join(target_list)
    multi_hop_rels = inter_direct_list + inter_direct_inter
    multi_hop_sentences = ','.join(multi_hop_rels)
    sep_1 = f"Direct paths from {source}:"
    sep2 = f"Direct paths from {target}:"
    sep3 = f"Paths between the nodes in the direct paths of {source} and {target}:"
    if additional_list:
        additional_sentences = ','.join(additional_list)
        sep4 = f"Additional relations related to the question"
        sentences = '\n'.join([sep_1, source_rels, sep2, target_rels, sep3, multi_hop_sentences, sep4, additional_sentences])
    else:
        sentences = '\n'.join([sep_1, source_rels, sep2, target_rels, sep3, multi_hop_sentences])
    answer = gen_chain.run(input=sentences, question=question, previous_answer=previous_answer)
    print(answer)
    return answer

class PharosGraphQA:
    def __init__(self, uri, username, password, llm, entity_types, additional_entity_types=None):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entity_types = entity_types
        self.additional_entity_types = additional_entity_types  # Store the additional entity types dictionary

    def _call(self, names_list, question, previous_answer, progress_callback=None):
        
        entities_list = list(self.entity_types.items())
        # The first entity is the source entity
        source_entity_name, source_entity_type = entities_list[0]
        # The second entity is the target entity
        target_entity_name, target_entity_type = entities_list[1]

        if source_entity_type == "Disease":
            source_result = disease_query(source_entity_name, question)
        elif source_entity_type == "Drug" or "Food" or "Metabolite":
            source_result = ligand_query(source_entity_name, question)
        elif source_entity_type == "Gene":
            source_result = target_query(source_entity_name, question)

        if source_result is not None:
            from_source_paths, from_source_nodes = result
        else:
            return None

        if target_entity_type == "Disease":
            target_result = disease_query(target_entity_name, question)
        elif target_entity_type == "Drug" or "Food" or "Metabolite":
            target_result = ligand_query(target_entity_name, question)
        elif target_entity_type == "Gene":
            target_result = target_query(target_entity_name, question)

        if target_result is not None:
            from_target_paths, from_target_nodes = result
        else:
            return None
            
        query_nodes = from_source_nodes + from_target_nodes
        query_nodes = set(list(query_nodes))

        if self.additional_entity_types is not None:
            additional_paths = set()
            for entityname, entity_info in self.additional_entity_types.items():
                if entity_info is not None and entity_info['entity_type'] is not None:  
                    entity_type = entity_info['entity_type'][1]  # Extract entity type from the nested dictionary
                    if entity_type == "Disease":
                        formatted_additional_entity_rels, additional_entity_nodes = disease_query(entityname, question)
                    elif entity_type == "Drug":
                        formatted_additional_entity_rels, additional_entity_nodes = ligand_query(entityname, question)
                    elif entity_type == "Gene":
                        formatted_additional_entity_rels, additional_entity_nodes = target_query(entityname, question)
                    
                    query_nodes.update(additional_entity_nodes)
                    additional_paths.update(formatted_additional_entity_rels)

        print("query nodes")
        print(len(query_nodes))
        print(query_nodes)

        mid_direct_paths = set()
        mid_direct_nodes = set()
        mid_direct_graph_rels = set()
        query_nodes = list(query_nodes)
        node_labels = get_node_labels_dict(self.graph, query_nodes)
        for node in query_nodes:
            paths = []  # Initialize paths to an empty list
            node_label = node_labels.get(node)
            if node_label is not None:
                paths = query_direct(self.graph, node, node_label)
            if paths:
                (selected_paths, 
                selected_nodes, 
                selected_graph_rels) = select_paths2(paths, 
                                                    question, 
                                                    max(1, len(paths)//3), 
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
        
        (selected_mid_inter_paths, 
         selected_mid_inter_nodes, 
         selected_mid_inter_graph_rels) = select_paths2(mid_inter_paths, 
                                                        question, 
                                                        len(mid_inter_paths)//3, 
                                                        30, 
                                                        progress_callback)
        

        print("final_inter_direct_inter_relationships")
        print(len(selected_mid_inter_paths))

        all_graph_rels = set()
        all_graph_rels.update(selected_mid_inter_graph_rels)
        all_graph_rels.update(mid_direct_graph_rels) 
        all_graph_rels.update(from_source_paths) 
        all_graph_rels.update(from_target_paths)
        
        if self.additional_entity_types is not None:
            all_graph_rels.update(formatted_additional_entity_rels)

        all_graph_rels = list(all_graph_rels)

        print("all_graph_rels")
        print(len(all_graph_rels))


########################################################################################################
                         
        params = {
            "llm": self.llm, 
            "question": question,
            "source_list": list(from_source_paths),
            "target_list": list(from_target_paths),
            "inter_direct_list": list(mid_direct_paths),
            "inter_direct_inter": list(selected_mid_inter_paths),
            "source": names_list[0],
            "target": names_list[1],
            "previous_answer": previous_answer
        }

        if self.additional_entity_types is not None:
            params["additional_rels"] = formatted_additional_entity_rels

        final_context = generate_answer(**params)

        answer = final_context

        response = {"result": answer, 
                    "all_rels": all_graph_rels}
                
        return response
