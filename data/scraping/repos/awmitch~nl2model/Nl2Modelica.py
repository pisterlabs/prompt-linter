from pydantic import BaseModel
from OMPython import OMCSessionZMQ, ModelicaSystem
from typing import Callable
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

class ModelObject():
    omc: OMCSessionZMQ
    lookup_chain: ConversationalRetrievalChain
    lookup_mem: ConversationBufferMemory
    modelica_system: ModelicaSystem = None
    nl2model_retriever: ConversationalRetrievalChain
    nl2model_vector: FAISS
    def __init__(
            self, 
            omc, 
            lookup_chain, 
            lookup_mem, 
            #modelica_system, 
            nl2model_retriever, 
            nl2model_vector,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.omc = omc
        self.lookup_chain = lookup_chain
        self.lookup_mem = lookup_mem
        #self.modelica_system = modelica_system
        self.nl2model_retriever = nl2model_retriever
        self.nl2model_vector = nl2model_vector
        
    class Config:
        arbitrary_types_allowed = True
    
    quantities = []
    continuous = []
    inputs = []
    outputs = []
    parameters = []
    simOptions = []
    solutions = []
    model_file = "model.mo"
    results_file = "results.mat"
    code = ""
    model_name = ""
    modelica_context = ""
    modelica_input = ""
    
    def get_value(self, query):
        list_mapping = {
            'quantities': self.quantities,
            'continuous': self.continuous,
            'inputs':     self.inputs,
            'outputs':    self.outputs,
            'parameters': self.parameters,
            'simOptions': self.simOptions,
            'solutions':  self.solutions,
            'code':       self.code
        }
        if query in list_mapping:
            return f"{list_mapping[query]}"
        else:
            return "Invalid attribute name"


