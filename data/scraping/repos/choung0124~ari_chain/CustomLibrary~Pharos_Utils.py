from CustomLibrary.Pharos_Queries import (
    query_ligand_targets,
    get_disease_targets_predictions,
    query_protein_protein_interactions,
    query_target_associated_diseases,
    query_disease_associated_targets
)
from langchain.llm import LLMChain
from langchain.prompts import PromptTemplate

class PharosGraphQA:
    def __init__(self, llm):
        self.llm = llm       

    def query_target_info(self, name):
        response, diseases_list, target_disease_rels = query_target_associated_diseases(name)
        response, protein_list, ppi_rels = query_protein_protein_interactions(name)
        
        return diseases_list, target_disease_rels, protein_list, ppi_rels
    
    def query_disease_info(self, name):
        response, target_list, disease_target_rels = query_disease_associated_targets(name)
        
        for target in target_list:
            response, diseases_list, target_disease_rels, protein_list, ppi_rels = self.query_target_info(target)

