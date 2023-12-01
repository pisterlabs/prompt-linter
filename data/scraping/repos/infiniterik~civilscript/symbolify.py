# Symbolic Stance Detection

import requests
from langchain.chains.base import Chain

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

stanceDescription = """A stance is a combination of a predicate expressed by the author, whether or not the author believes said predicate, and the author's sentiment towards the predicate."""

abortion_belief_types = """'CHOOSE_LIFE', 'EXPERIENCE_PAIN', 'STRUGGLE', 'FREEDOM_OF_CHOICE', 'SUPPORT', 'REGRET', 'RUIN', 'AFFORD'"""

import json
import sys, os
from typing import Dict, List

llm = OpenAI(temperature=0.9)
oldPredicatetemplate=stanceDescription+"""
    Consider the following comment and explanation regarding stances about {domain} the text expresses. 
    What is the main predicate that stances refer to? The predicate is represented as a verb and the main argument of the verb in the form VERB[ARGUMENT].
    Return the response in the form "BELIEF_TYPE[PREDICATE]" and use the minimum number of words necessary to uniquely identify the predicate.
    For abortion, the only allowable belief types are: """+abortion_belief_types+""". 
    Remember that there may be multiple stances. Return a separate predicate representation for each stance separated by commas.
    Comment:{text}
    Explanation: {explanation}
    Predicate:"""

getPredicate = PromptTemplate(
    input_variables=["text", "explanation", "domain"],
    template=stanceDescription+"""
    Consider the following comment and explanation regarding stances about {domain} the text expresses. 
    What is the main predicate that stances refer to? The predicate is represented as a verb and the main argument of the verb in the form VERB[ARGUMENT].
    Use the minimum number of words necessary to uniquely identify the predicate but remember that all the terms from the predicate must be in the original comment.
    Remember that there may be multiple stances. Return a separate predicate representation for each stance separated by commas.
    Explanation: {explanation}
    Comment:{text}
    Predicate:""",
)

getPredicateChain = LLMChain(llm=llm, prompt=getPredicate, output_key="predicate")

getBeliefType = PromptTemplate(
    input_variables=["text", "explanation", "domain", "predicate"],
    template=stanceDescription+"""Consider the following predicate extracted from the comment and explanation regarding stances about {domain} the comment expresses.
    What is the belief type of the predicate? Respond with one of the following:
    """+abortion_belief_types+"""
    You must respond with one of the above terms. Ensure that only one of the above terms is used as the response.
    Explanation: {explanation}
    Comment:{text}
    Predicate: {predicate}
    Belief Type:""")

getBeliefTypeChain = LLMChain(llm=llm, prompt=getBeliefType, output_key="belief_type")

getSentiment = PromptTemplate(
    input_variables=["text", "belief_type", "predicate", "explanation", "domain"],
    template=stanceDescription+"""
    Consider the following comment and explanation regarding stances about {domain} the text expresses. 
    What is the sentiment of the author towards the stance predicate {belief_type}[{predicate}]? Respond with one of the following:
    - Positive
    - Negative
    - Neutral
    Explanation: {explanation}
    Comment:{text}
    Sentiment:""",
)

getSentimentChain = LLMChain(llm=llm, prompt=getSentiment, output_key="sentiment")

getBelief = PromptTemplate(
    input_variables=["text", "belief_type", "predicate", "explanation", "domain"],
    template=stanceDescription+"""
    Consider the following comment and explanation regarding stances about {domain} the text expresses. 
    How strongly does the author believe the stance predicate {belief_type}[{predicate}]? Respond with one of the following:
    - Very strongly believes
    - Strongly believes
    - Believes
    - Does not believe
    - Strongly does not believe
    - Very strongly does not believe
    Ensure that only one of the above terms is used as the response.
    Explanation: {explanation}
    Comment:{text}
    Belief Strength:""",
)

getBeliefChain = LLMChain(llm=llm, prompt=getBelief, output_key="belief")

class SymbolicExtractor(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ['text', 'domain', 'explanation']

    @property
    def output_keys(self) -> List[str]:
        return ['stances']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        predicates = getPredicateChain(inputs)
        result = []
        for p in predicates["predicate"].split(","):
            i = dict()
            inputs["predicate"] = p
            i["predicate"] = p
            i["belief_type"] = getBeliefTypeChain(inputs)["belief_type"]
            inputs["belief_type"] = i["belief_type"]
            i["sentiment"] = getSentimentChain(inputs)["sentiment"]
            i["belief"] = getBeliefChain(inputs)["belief"]
            result.append(i)
        return {"stances": result}

SymbolifyChain2 = SequentialChain(chains=[getPredicateChain, getBeliefTypeChain, getSentimentChain, getBeliefChain],
                                                input_variables=["text", "explanation", "domain"],
                                                output_variables=["sentiment", "belief", "belief_type", "predicate"],
                                                verbose=True)

SymbolifyChain = SymbolicExtractor()