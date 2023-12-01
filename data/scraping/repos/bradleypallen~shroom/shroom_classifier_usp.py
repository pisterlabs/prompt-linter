# -*- coding: utf-8 -*-

from datetime import datetime
from langchain import HuggingFaceHub, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class ShroomClassifier:
    """Represents a classifier for the SHROOM evaluation dataset."""

    TASK = {
        "DM": "The given task is Definition Modeling, where the goal is to generate a definition for the term between the '<define>' and '</define>' delimiters in the input.",
        "PG": "The given task is Paraphrase Generation, where the goal is to generate a paraphrase of the input.",
        "MT": "The given task is Machine Translation, where the goal is to generate a natural language translation of the input.",
        "TS": "The given task is Text Simplification, where the goal is to generate a simplified version of the input.",
    }

    REFERENCE = {
        "src": "the input",
        "tgt": "the target",
        "either": "either the input or the target",
    }

    RATIONALE_GENERATION_PROMPT = """{task}  
You have been provided with the below inputs, outputs and targets. Your goal is to determine if the output is 
a hallucination, defined as an output that contains information that is not supported by the reference. Using {ref} 
as the reference, provide a succinct rationale arguing for or against the assertion that the output is a hallucination.
##
Input: {src}
Target: {tgt} 
Output: {hyp}
Rationale:
"""

    ANSWER_GENERATION_PROMPT = """Using the argument provided in the below rationale, answer the question: 
is the output a hallucination? Answer 'Hallucination' if the output is a hallucination, or 'Not Hallucination'  
if the output is not a hallucination. Only answer 'Hallucination' or 'Not Hallucination'. 
##
Input: {src}
Target: {tgt} 
Output: {hyp}
Rationale: {rationale}
Answer:
"""
    
    def __init__(self, model_name="gpt-4", temperature=0.1):
        """
        Initializes a classifier for the SemEval 2024 Task 6, "SHROOM, a Shared-task on Hallucinations and Related Observable Overgeneration Mistakes".
        
        Parameters:
            model_name: The name of the model to be used for zero shot CoT classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._llm(model_name, temperature)
        self._classify_chain = self._zero_shot_chain_of_thought()

    def _llm(self, model_name, temperature):
        if model_name in [
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4-1106-preview",
            ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature, request_timeout=50)
        elif model_name in [
            "text-curie-001"
            ]:
            return OpenAI(model_name=model_name, temperature=temperature, request_timeout=50)
        elif model_name in [
            "meta-llama/Llama-2-70b-chat-hf", 
            "google/flan-t5-xxl",
            ]:
            return HuggingFaceHub(repo_id=model_name, model_kwargs={ "temperature": temperature })
        else:
            raise Exception(f'Model {model_name} not supported')

    def _zero_shot_chain_of_thought(self):
        """
        Creates a langchain.SequentialChain that implements a zero-shot
        chain of thought (CoT) using a specification. 
        """
        rationale_generation = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=[ "task", "src", "tgt", "hyp", "ref"], 
                template=self.RATIONALE_GENERATION_PROMPT
            ), 
            output_key="rationale"
        )
        answer_generation = LLMChain(
            llm=self.llm, 
            prompt=PromptTemplate(
                input_variables=["src", "tgt", "hyp", "rationale"], 
                template=self.ANSWER_GENERATION_PROMPT
            ), 
            output_key="label"
        )
        return SequentialChain(
            chains=[rationale_generation, answer_generation],
            input_variables=["task", "src", "tgt", "hyp", "ref"],
            output_variables=["rationale", "label"]
        )
    
    def classify(self, task, src, tgt, hyp, ref):
        """
        Determines whether or not the output (hyp) is a hallucination.
        
        Parameters:
            task: The task associated with a datapoint. One of "DM", "PG", "MT", or "TS".
            src: The input passed to a model.
            tgt: The intended reference "gold" text that the model ought to generate.
            hyp: The output the model generated.
            ref: The field(s) containing the semantic information used to determine if the input is a hallucination. One of "src", "tgt", or "either".
       
        Returns:
            A dict containing a classification of the output based on the task, reference, input, output and target.
        """
        classifications = [ 
            self._classify_chain({ 
                "task": self.TASK[task], 
                "src": src, 
                "tgt": tgt, 
                "hyp": hyp,
                "ref": self.REFERENCE[ref],
            }) for i in range(5) 
        ]
        predictions = [ classification["label"] for classification in classifications ]
        weight = 1./float(len(predictions))
        predicted_p = float(sum([ weight for prediction in predictions if prediction == 'Hallucination' ]))
        if predicted_p > 0.5:
            predicted = "Hallucination"
        else:
            predicted = "Not Hallucination"
        output = {
            "predictions": predictions,
            "predicted": predicted,
            "predicted_p": predicted_p,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + 'Z'
        }
        for i, classification in enumerate(classifications):
            output[f'rationale_{i}'] = classification["rationale"]
        return output
