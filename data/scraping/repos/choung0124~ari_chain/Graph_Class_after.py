from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.chains.llm import LLMChain
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from py2neo import Graph
import numpy as np
from CustomLibrary.Graph_Queries import (
    find_shortest_paths, 
    query_direct, 
    query_inter_relationships_between_direct
)
from CustomLibrary.Graph_Utils import (
    select_paths2, 
    generate_answer, 
)

class KnowledgeGraphRetrieval:
    def __init__(self, uri, username, password, llm, entity_types, additional_entity_types=None):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entity_types = entity_types
        self.additional_entity_types = additional_entity_types  # Store the additional entity types dictionary

    def _call(self, names_list, question, progress_callback=None):

        (source_to_target_paths, 
         from_target_paths, 
         from_source_paths, ) = find_shortest_paths(self.graph, 
                                                  names_list, 
                                                  self.entity_types, 
                                                  repeat=True)
        
        (selected_from_target_paths, 
         selected_from_target_nodes, 
         selected_from_target_graph_rels) = select_paths2(from_target_paths, 
                                                          question, 
                                                          len(from_target_paths)//15, 
                                                          10, 
                                                          progress_callback)
        
        print("final_target_paths")
        print(len(selected_from_target_paths))

        (selected_from_source_paths, 
         selected_from_source_nodes, 
         selected_from_source_graph_rels) = select_paths2(from_source_paths, 
                                                          question, 
                                                          len(from_source_paths)//15,
                                                          10, 
                                                          progress_callback)
        
        print("final_source_paths")
        print(len(selected_from_source_paths))

        (selected_source_to_target_paths, 
         selected_source_to_target_nodes, 
         selected_source_to_target_graph_rels) = select_paths2(source_to_target_paths, 
                                                               question, 
                                                               len(source_to_target_paths), 
                                                               10, 
                                                               progress_callback)
        
        print("final_inter_relationships")
        print(len(selected_source_to_target_paths))


        if self.extraEntityTypes is not None:
            additional_nodes = set()
            additional_paths = set()
            additional_graph_rels = set()

            for entityName, entityInfo in self.extraEntityTypes.items():
                paths = query_direct(self.graph, entityName)
                selected_paths, selected_nodes, selected_graph_rels = select_paths2(paths, 
                                                                                    question, 
                                                                                    len(paths)//15,
                                                                                    10, 
                                                                                    progress_callback)
                
                additional_paths.update(selected_paths)
                additional_graph_rels.update(selected_graph_rels)
                additional_nodes.update(selected_nodes)

            print("additional_entity_direct_graph_relationships")
            print(additional_paths)

        query_nodes = selected_from_target_nodes + selected_from_source_nodes + selected_source_to_target_nodes
        
        if self.extraEntityTypes is not None:
            query_nodes += list(additional_nodes)

        query_nodes = (set(query_nodes))
        names_set = set(names_list)

        query_nodes = [name for name in query_nodes if name.lower() not in names_set]
        print("query nodes")
        print(len(query_nodes))
        print(query_nodes)

        mid_direct_paths = set()
        mid_direct_nodes = set()
        mid_direct_graph_rels = set()
        
        for node in query_nodes:
            paths = query_direct(self.graph, node)
            if paths:
                (selected_paths, 
                 selected_nodes, 
                 selected_graph_rels) = select_paths2(paths, 
                                                      question, 
                                                      len(paths)//15, 
                                                      3, 
                                                      progress_callback)
                
                mid_direct_paths.update(selected_paths)
                mid_direct_nodes.update(selected_nodes)
                mid_direct_graph_rels.update(selected_graph_rels)
                
                print("success")
                print(len(mid_direct_paths))
                print(mid_direct_paths)
            else:
                print("skipping")
                continue

        print("number of unique inter_direct_relationships:")
        print(len(mid_direct_paths))

        mid_inter_paths = query_inter_relationships_between_direct(self.graph, 
                                                                   list(mid_direct_nodes), 
                                                                   query_nodes)
        
        (selected_mid_inter_paths, 
         selected_mid_inter_nodes, 
         selected_mid_inter_graph_rels) = select_paths2(mid_inter_paths, 
                                                        question, 
                                                        len(mid_inter_paths)//15, 
                                                        50, 
                                                        progress_callback)
        

        print("final_inter_direct_inter_relationships")
        print(len(selected_mid_inter_paths))

        all_graph_rels = (selected_mid_inter_graph_rels + 
                          mid_direct_graph_rels + 
                          selected_source_to_target_graph_rels + 
                          selected_from_source_graph_rels + 
                          selected_from_target_graph_rels)
        if self.additional_entity_types is not None:
            all_graph_rels += additional_graph_rels

        all_graph_rels = list(set(all_graph_rels))

        print("all_graph_rels")
        print(len(all_graph_rels))

########################################################################################################
        
        params = {
            "llm": self.llm, 
            "relationships_list": source_to_target_paths,
            "question": question,
            "source_list": from_source_paths,
            "target_list": from_target_paths,
            "inter_direct_list": mid_direct_paths,
            "inter_direct_inter": mid_inter_paths,
            "source": names_list[0],
            "target": names_list[1]
        }

        if self.additional_entity_types is not None:
            params["additional_rels"] = additional_graph_rels

        final_context = generate_answer(**params)

        answer = final_context

        response = {"result": answer, 
                    "all_rels": all_graph_rels}
                
        return response


