# -*- coding: utf-8 -*-

from datetime import datetime
from langchain import HuggingFaceHub, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class ShroomEnsembleClassifier:
    """Represents a classifier for the SHROOM evaluation dataset."""

    PERSONA = {
        "translator": "a translator concerned that the output is a good translation.",
        "editor": "an editor concerned that the output is understandable.",
        "writer": "a creative writer concerned that the output is engaging.",
        "grammarian": "a grammarian concerned that the output is grammatical.",
        "lawyer": "a lawyer concerned that the output is truthful.",
    }

    TASK = {
        "DM": "The given task is Definition Modeling, meaning that the goal of the language model is to generate a definition for the term between the '<define>' and '</define>' delimiters in the input.",
        "PG": "The given task is Paraphrase Generation, meaning that the goal of the language model is to generate a paraphrase of the input.",
        "MT": "The given task is Machine Translation, meaning that the goal of the language model is to generate a natural language translation of the input.",
        "TS": "The given task is Text Simplification, meaning that the goal of the language model is to generate a simplified version of the input.",
    }

    RATIONALE_GENERATION_PROMPT = """Input: {src}
Target: {tgt} 
Output: {hyp}
Persona: {persona}

{task}  
You have been provided with the above input and output pair, as well as a target that you need to use 
to determine if the output is correct and accurate, or if it is a hallucination, defined as an output 
that is incorrect, off point, or contains extraneous information that cannot be reasonably inferred from the input. 
Provide a succinct rationale arguing for or against the assertion that the output is a hallucination, 
based on your expertise as {persona_desc}, restricting yourself to judgments solely within your expertise.

Rationale:
"""

    ANSWER_GENERATION_PROMPT = """Input: {src}
Target: {tgt} 
Output: {hyp}
Persona: {persona}
Rationale: {rationale}

Now using the argument provided in the above rationale, answer the question: is the output a hallucination? 
Answer 'Hallucination' if the output is a hallucination, or 'Not Hallucination' is not a hallucination. Only answer 
'Hallucination' or 'Not Hallucination'.
  
Answer:
"""
    
    def __init__(self, model_name="gpt-4", temperature=0.1):
        """
        Initializes a classifier for the SemEval 2024 Task 6, "".
        
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
            return ChatOpenAI(model_name=model_name, temperature=temperature, request_timeout=100)
        elif model_name in [
            "text-curie-001"
            ]:
            return OpenAI(model_name=model_name, temperature=temperature, request_timeout=100)
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
                input_variables=[ "task", "persona", "persona_desc", "src", "tgt", "hyp"], 
                template=self.RATIONALE_GENERATION_PROMPT
            ), 
            output_key="rationale"
        )
        answer_generation = LLMChain(
            llm=self.llm, 
            prompt=PromptTemplate(
                input_variables=["src", "tgt", "hyp", "persona", "rationale"], 
                template=self.ANSWER_GENERATION_PROMPT
            ), 
            output_key="label"
        )
        return SequentialChain(
            chains=[rationale_generation, answer_generation],
            input_variables=["task", "persona", "persona_desc", "src", "tgt", "hyp"],
            output_variables=["persona", "rationale", "label"]
        )
    
    def classify(self, task, src, tgt, hyp):
        """
        Determines whether or not the output (hyp) is a hallucination.
        
        Parameters:
            task: The task associated with a datapoint. One of "DM", "PG", "MT", or "TS".
            persona: The name of the persona used by the LM in generating the rationale.
            persona_desc: The description of the persona.
            src: The input passed to a model.
            tgt: The intended reference "gold" text that the model ought to generate
            hyp: The output the model generated.
        
        Returns:
            A dict containing a classification of the output based on the task, persona, input, output and target.
        """
        classifications = [ 
            self._classify_chain({ 
                "task": self.TASK[task], 
                "persona": persona, 
                "persona_desc": self.PERSONA[persona], 
                "src": src, 
                "tgt": tgt, 
                "hyp": hyp 
            }) for persona in self.PERSONA 
        ]
        predictions = [ classification["label"] for classification in classifications ]
        weight = 1./float(len(predictions))
        rationale = [ classification["rationale"] for classification in classifications ]
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
        for classification in classifications:
            output[f'{classification["persona"]}_r'] = classification["rationale"]
        return output
