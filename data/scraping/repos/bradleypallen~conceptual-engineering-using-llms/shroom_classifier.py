import yaml
from langchain import HuggingFaceHub, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class ShroomClassifier:
    """Represents a classifier for the SHROOM evaluation dataset."""

    DM_RATIONALE_GENERATION_PROMPT = """Input: {src}
Target: {tgt} 
Output: {hyp}
  
You are a data quality engineer verifying that a language model is generating correct and accurate output from a given input.
The given task is Definition Modeling, meaning that the goal of the language model is to generate a definition for the term between
the '<define>' and '</define>' delimiters in the input. You have been provided with the above input and output pair, as well as a target 
that you need to use to determine if the output is correct and accurate, or if it is a hallucination, defined as an output that is 
incorrect, off point, or contains information that cannot be reasonably inferred from the input. 
Provide a rationale arguing for or against the assertion that the output is a hallucination.
    
Rationale:
"""

    PG_RATIONALE_GENERATION_PROMPT = """Input: {src}
Target: {tgt} 
Output: {hyp}
  
You are a data quality engineer verifying that a language model is generating correct and accurate output from a given input.
The given task is Paraphrase Generation, meaning that the goal of the language model is to generate a paraphrase of the input. 
You have been provided with the above input and output pair, as well as a target 
that you need to use to determine if the output is correct and accurate, or if it is a hallucination, defined as an output that is 
incorrect, off point, or contains information that cannot be reasonably inferred from the input. 
Provide a rationale arguing for or against the assertion that the output is a hallucination.
    
Rationale:
"""

    MT_RATIONALE_GENERATION_PROMPT = """Input: {src}
Target: {tgt} 
Output: {hyp}
  
You are a data quality engineer verifying that a language model is generating correct and accurate output from a given input.
The given task is Machine Translation, meaning that the goal of the language model is to generate a natural language translation 
of the input. You have been provided with the above input and output pair, as well as a target 
that you need to use to determine if the output is correct and accurate, or if it is a hallucination, defined as an output that is 
incorrect, off point, or contains information that cannot be reasonably inferred from the input. 
Provide a rationale arguing for or against the assertion that the output is a hallucination.
    
Rationale:
"""

    TS_RATIONALE_GENERATION_PROMPT = """Input: {src}
Target: {tgt} 
Output: {hyp}
  
You are a data quality engineer verifying that a language model is generating correct and accurate output from a given input.
The given task is Definition Modeling, meaning that the goal of the language model is to generate a simplified version of the input. 
You have been provided with the above input and output pair, as well as a target 
that you need to use to determine if the output is correct and accurate, or if it is a hallucination, defined as an output that is 
incorrect, off point, or contains information that cannot be reasonably inferred from the input. 
Provide a rationale arguing for or against the assertion that the output is a hallucination.
    
Rationale:
"""

    ANSWER_GENERATION_PROMPT = """Input: {src}
Target: {tgt} 
Output: {hyp}
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
        self._DM_classify_chain = self._zero_shot_chain_of_thought(self.DM_RATIONALE_GENERATION_PROMPT)
        self._PG_classify_chain = self._zero_shot_chain_of_thought(self.PG_RATIONALE_GENERATION_PROMPT)
        self._MT_classify_chain = self._zero_shot_chain_of_thought(self.MT_RATIONALE_GENERATION_PROMPT)
        self._TS_classify_chain = self._zero_shot_chain_of_thought(self.TS_RATIONALE_GENERATION_PROMPT)

    def _llm(self, model_name, temperature):
        if model_name in [
            "gpt-4",
            "gpt-3.5-turbo",
            ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature)
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

    def _zero_shot_chain_of_thought(self, rationale_generation_prompt_template):
        """
        Creates a langchain.SequentialChain that implements a zero-shot
        chain of thought (CoT) using a specification. 

        Parameters:
            rationale_generation_prompt_template: The prompt template for rationales for the given task.
        """
        rationale_generation = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["src", "tgt", "hyp"], 
                template=rationale_generation_prompt_template
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
            input_variables=["src", "tgt", "hyp"],
            output_variables=["rationale", "label"]
        )
    
    def classify(self, task, src, tgt, hyp):
        """
        Determines whether or not the output (hyp) is a hallucination.
        
        Parameters:
            src: The input passed to a model.
            tgt: The intended reference "gold" text that the model ought to generate
            hyp: The output the model generated.
        
        Returns:
            A JSON object containing a classification of the output based on the input, output and target.
        """
        if task == "DM":
            return self._DM_classify_chain({ "src": src, "tgt": tgt, "hyp": hyp })
        elif task == "PG":
            return self._PG_classify_chain({ "src": src, "tgt": tgt, "hyp": hyp })
        elif task == "MT":
            return self._MT_classify_chain({ "src": src, "tgt": tgt, "hyp": hyp })
        else:
            return self._TS_classify_chain({ "src": src, "tgt": tgt, "hyp": hyp })
