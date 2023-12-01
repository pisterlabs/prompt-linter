from typing import List, Dict, Any, Optional
import random
import numpy as np

#----------------------------------------------------------------------------------
"""Node that run map-reduce on opinions"""
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

from langchain.chains import ReduceDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import pdb
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# pdb.set_trace()
import sys
import os
sys.path.append(".")
from opinion_networks.reasoner import Reasoner
from opinion_networks.engine import Value

from opinion_networks.opinion import OpinionPair

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool = True, language: str = 'Spanish', background_fn: Optional = None, **kwargs):
        #assert len(opinions) == nin, "There should the same number of weights and opinions"
        # 2*nin because we have to store the attention of each neuron on each positive and negative opinions
        self.w_pos = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.w_neg = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b_pos = Value(random.uniform(-1,1))
        self.b_neg = Value(random.uniform(-1,1))
        self.nonlin = nonlin
        self.opinions_in: list[str] = None
        self.opinion_out: str = None
        self.background: str = background_fn()
        self.language: str = language
        self.type = kwargs['type']
        self.temperature = 0.0 
        #self.temperature = random.uniform(0.0, 1.0) #sometimes it doesn;t generate valid JSON files
        self.reasoner = Reasoner(
            background=self.background, 
            language=self.language,
            temperature=self.temperature, 
            **kwargs
        )

    def __call__(self, opinions: Any) -> OpinionPair:
        self.opinions_in = opinions
        opinions = self.reasoner(opinions, self.background)
        
        if self.type != 'input':
            pos_scores = [opinions.pos_opinion.score for opinions in self.opinions_in]
            act_pos = sum((wi*xi for wi, xi in zip(self.w_pos, pos_scores)), self.b_pos)
            act_pos = act_pos.relu() if self.nonlin else act_pos
            opinions.pos_opinion.score = act_pos

            neg_scores = [out.neg_opinion.score for out in self.opinions_in]
            act_neg = sum((wi*xi for wi, xi in zip(self.w_neg, neg_scores)), self.b_neg)
            act_neg = act_neg.relu() if self.nonlin else act_neg
            opinions.neg_opinion.score = act_neg
        return opinions

    def parameters(self):
        return self.w_pos + [self.b_pos] + self.w_neg + [self.b_neg]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({2*len(self.w_pos)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        
    def __call__(self, opinions: list[OpinionPair]) -> list[OpinionPair]:
        opinions = [n(opinions) for n in self.neurons]
        # x = [o[0] for o in out]
        # opinion = [o[1] for o in out]
        
        # x = x[0] if len(x) == 1 else x
        #opinion = opinion[0] if len(opinion) == 1 else opinion
        return opinions

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts, llm=None, **kwargs):
        sz = [nin] + nouts
        self.layers = []
        self.opinions = []
        for i in range(len(nouts)):
            if i == 0:
                self.layers.append(Layer(sz[i], sz[i+1], type='input', **kwargs))
            elif i == len(nouts)-1:
                self.layers.append(Layer(sz[i], sz[i+1], type='output', **kwargs))
            else:
                self.layers.append(Layer(sz[i], sz[i+1], type='hidden', **kwargs))

    def __call__(self,  doc):
        opinions = doc
        for layer_i, layer in enumerate(self.layers):
            opinions = layer(opinions)
            self.opinions.append({'layer_id': layer_i, 'opinions': opinions})
        return opinions

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"