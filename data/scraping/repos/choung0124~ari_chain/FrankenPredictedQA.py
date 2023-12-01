from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.chains.llm import LLMChain
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from py2neo import Graph
import numpy as np
from langchain.prompts import PromptTemplate

from CustomLibrary.Graph_Queries import (
    query_inter_relationships_direct1, 
    query_inter_relationships_between_direct
    )

from CustomLibrary.Graph_Utils import (
    select_paths, 
    select_paths2, 
)
from CustomLibrary.Custom_Prompts import Graph_Answer_Gen_Template3

from CustomLibrary.OpenTargets import(
    query_predicted_disease_info,
    query_predicted_target_info,
    query_predicted_drug_info
)

def generate_answer(llm, source_list, target_list, inter_direct_list, inter_direct_inter, question, previous_answer, source, target, additional_list:Optional[List[str]]=None):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template3, input_variables=["input", "question", "previous_answer"])
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
        sep4 = f"Direct paths from "
        sentences = '\n'.join([sep_1, source_rels, sep2, target_rels, sep3, multi_hop_sentences, sep4, additional_sentences])
    else:
        sentences = '\n'.join([sep_1, source_rels, sep2, target_rels, sep3, multi_hop_sentences])
    answer = gen_chain.run(input=sentences, question=question, previous_answer=previous_answer)
    print(answer)
    return answer

class FrankenPredictedQA:
    def __init__(self, uri, username, password, llm, entity_types, question, additional_entity_types=None):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entity_types = entity_types
        self.additional_entity_types = additional_entity_types  # Store the additional entity types dictionary
        entities_list = list(self.entity_types.items())

        # The first entity is the source entity
        source_entity_name, source_entity_type = entities_list[0]

        # The second entity is the target entity
        target_entity_name, target_entity_type = entities_list[1]

        self.source_result_dict = {}
        self.target_result_dict = {}

        if source_entity_type == "Disease":
            self.source_result_dict = query_predicted_disease_info(source_entity_name, question)
            self.source_similar_entity_names = list(self.source_result_dict.keys())
        if source_entity_type == "Drug":
            self.source_result_dict = query_predicted_drug_info(source_entity_name, question)
            self.source_similar_entity_names = list(self.source_result_dict.keys())
        if source_entity_type == "Gene":
            self.source_result_dict = query_predicted_target_info(source_entity_name, question)
            self.source_similar_entity_names = list(self.source_result_dict.keys())

        if target_entity_type == "Disease":
            self.target_result_dict = query_predicted_disease_info(target_entity_name, question)
            self.target_similar_entity_names = list(self.target_result_dict.keys())
        if target_entity_type == "Drug":
            self.target_result_dict = query_predicted_drug_info(target_entity_name, question)
            self.target_similar_entity_names = list(self.target_result_dict.keys())
            print(self.target_similar_entity_names)
        if target_entity_type == "Gene":
            self.target_result_dict = query_predicted_target_info(target_entity_name, question)
            self.target_similar_entity_names = list(self.target_result_dict.keys())
        
        # Initialize dictionaries for storing result dictionaries and key lists
        self.additional_result_dicts = {}
        self.additional_similar_entity_names_lists = {}
        self.additional_similar_entity_names = {}  # Add this line

        if self.additional_entity_types is not None:
            for entityname, entity_info in self.additional_entity_types.items():
                if entity_info is not None and entity_info['entity_type'] is not None:  
                    entity_type = entity_info['entity_type'][1]  # Extract entity type from the nested dictionary
                    if entity_type == "Disease":
                        self.additional_result_dict = query_predicted_disease_info(entityname, question)
                    elif entity_type == "Drug":
                        self.additional_result_dict = query_predicted_drug_info(entityname, question)
                    elif entity_type == "Gene":
                        self.additional_result_dict = query_predicted_target_info(entityname, question)
                    
                    # Store the result dictionary and list of keys for this additional entity
                    self.additional_result_dicts[entityname] = self.additional_result_dict
                    self.additional_similar_entity_names_lists[entityname] = list(self.additional_result_dict.keys())


    def _call(self, names_list, question, previous_answer, generate_an_answer, progress_callback=None):
        # Get the maximum length among all lists of keys
        max_length = max(len(self.source_similar_entity_names), len(self.target_similar_entity_names), max([len(names) for names in self.additional_similar_entity_names_lists.values()]))

        # Iterate over the range of the maximum length
        for i in range(max_length):
            source_nodes = source_rels = target_nodes = target_rels = []
            if i < len(self.source_similar_entity_names) and i < len(self.target_similar_entity_names):
                print(self.target_similar_entity_names)
                names_to_print = [self.source_similar_entity_names[i], self.target_similar_entity_names[i]]
                name = self.source_similar_entity_names[i]
                source_paths = self.source_result_dict[name]['paths']
                source_nodes = self.source_result_dict[name]['nodes']
                source_rels = self.source_result_dict[name]['rels']
                name = self.target_similar_entity_names[i]
                target_paths = self.target_result_dict[name]['paths']
                target_nodes = self.target_result_dict[name]['nodes']
                target_rels = self.target_result_dict[name]['rels']
            
            # Initialize query_nodes, graph_rels and final_additional_paths with source and target nodes and relationships
            query_nodes = source_nodes + target_nodes
            graph_rels = source_rels + target_rels
            final_additional_paths = []

            if self.additional_entity_types is not None:
                for entityname in self.additional_entity_types:
                    additional_similar_entity_names = self.additional_similar_entity_names_lists[entityname]

                    if i < len(additional_similar_entity_names):
                        name = additional_similar_entity_names[i]
                        additional_result_dict = self.additional_result_dicts[entityname]
                        additional_paths = additional_result_dict[name]['paths']
                        additional_nodes = additional_result_dict[name]['nodes']
                        additional_rels = additional_result_dict[name]['rels']
                        names_to_print.append(name)
                    else:
                        names_to_print.append(None)  # Append None or some default value when no more entities are available

                    # Add the nodes and relationships of this additional entity to the query_nodes and graph_rels
                    query_nodes += additional_nodes
                    graph_rels += additional_rels
                    final_additional_paths += additional_paths

            # Remove duplicates by converting to sets
            query_nodes = set(query_nodes)
            graph_rels = set(graph_rels)
            names_set = set(names_list)
            #query_nodes.update(final_path_nodes)
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
                    inter_direct_relationships, selected_nodes, inter_direct_unique_rels, selected_target_direct_paths = select_paths(target_direct_relations, question, 15, 2, progress_callback)
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
        
            final_inter_direct_relationships = list(og_target_direct_relations)
            final_selected_inter_direct_nodes = list(set(selected_inter_direct_nodes))
            final_inter_direct_unique_graph_rels = list(set(inter_direct_unique_graph_rels))
            print("number of unique inter_direct_relationships:")
            print(len(final_inter_direct_relationships))

            if final_inter_direct_relationships:
                target_inter_relations, inter_direct_inter_unique_graph_rels, source_and_target_nodes2 = query_inter_relationships_between_direct(self.graph, final_selected_inter_direct_nodes, query_nodes)
                final_inter_direct_inter_relationships, selected_inter_direct_inter_nodes, inter_direct_inter_unique_rels = select_paths2(target_inter_relations, question, 15, 30, progress_callback)
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
            all_unique_graph_rels.update(graph_rels)
            all_unique_graph_rels.update(final_inter_direct_unique_graph_rels)
            all_unique_graph_rels.update(inter_direct_inter_unique_rels)


########################################################################################################
        
            if generate_an_answer and self.additional_entity_types is not None:
                final_context = generate_answer(
                    llm=self.llm, 
                    question=question,
                    previous_answer=previous_answer,
                    source_list=source_paths, 
                    target_list=target_paths,
                    inter_direct_list=final_inter_direct_relationships,
                    inter_direct_inter=final_inter_direct_inter_relationships,
                    source=names_list[0],
                    target=names_list[1],
                    additional_list=additional_paths,
                )
            else:
                final_context = generate_answer(
                    llm=self.llm, 
                    question=question,
                    previous_answer=previous_answer,
                    source_list=source_paths, 
                    target_list=target_paths,
                    inter_direct_list=final_inter_direct_relationships,
                    inter_direct_inter=final_inter_direct_inter_relationships,
                    source=names_list[0],
                    target=names_list[1]
                )

            # Update the previous_answer with the final_context for the next iteration
            previous_answer = final_context

            answer = final_context

            response = {
                "result": answer,
                "names_to_print": names_to_print,
                "multi_hop_relationships": final_inter_direct_inter_relationships,
                "source_relationships": source_paths,
                "target_relationships": target_paths,
                "inter_direct_relationships": final_inter_direct_relationships,
                "inter_direct_inter_relationships": final_inter_direct_inter_relationships,
                "all_nodes": all_nodes,
                "all_rels": all_unique_graph_rels
            }
            
            yield response