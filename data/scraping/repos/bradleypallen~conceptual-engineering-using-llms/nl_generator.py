import yaml
from langchain import HuggingFaceHub, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class NaturalLanguageDescriptionGenerator:
    """Represents a natural language generator that translates an RDF serialization into a set of natural language statements."""
    
    def __init__(self, model_name="gpt-4", temperature=0.1):
        """
        Initializes a NL generator.
        
        Parameters:
            model_name: The name of the model to be used for zero shot CoT classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._llm(model_name, temperature)
        self._generator_chain = self._zero_shot_chain_of_thought('./chains/generate_natural_language_description.yaml')

    def to_json(self):
        """
        Converts the generator to a JSON-like dictionary format.
        
        Returns:
            A dictionary with keys "id", "label", and "definition".
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,           
        }
    
    def _llm(self, model_name, temperature):
        if model_name in [
            "gpt-4",
            "gpt-3.5-turbo",
            ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature)
        elif model_name in [
            "text-curie-001"
            ]:
            return OpenAI(model_name=model_name, temperature=temperature)
        elif model_name in [
            "meta-llama/Llama-2-70b-chat-hf", 
            "google/flan-t5-xxl",
            ]:
            return HuggingFaceHub(repo_id=model_name, model_kwargs={ "temperature": temperature })
        else:
            raise Exception(f'Model {model_name} not supported')

    def _zero_shot_chain_of_thought(self, file):
        """
        Creates a langchain.SequentialChain that implements a zero-shot
        chain of thought (CoT) using a specification.
        
        Parameters:
            file: The name of the YAML file containing a specification of the CoT.
        """
        chain_specification = yaml.safe_load(open(file, 'r'))
        template_1 = chain_specification["rationale_generation"]
        chain_1 = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=template_1["input_variables"], 
                template=template_1["template"]
            ), 
            output_key=template_1["output_key"]
        )
        template_2 = chain_specification["answer_generation"]
        chain_2 = LLMChain(
            llm=self.llm, 
            prompt=PromptTemplate(
                input_variables=template_2["input_variables"], 
                template=template_2["template"]
            ), 
            output_key=template_2["output_key"]
        )
        return SequentialChain(
            chains=[chain_1, chain_2],
            input_variables=template_1["input_variables"],
            output_variables=chain_specification["output_variables"]
        )
    
    def describe(self, entity_label, entity_serialization):
        """
        Generates a natural language description of an entity based on its RDF serialization.
        
        Parameters:
            entity_label: The label of the entity to be classified.
            entity_serialization: The RDF serialization of the entity to be classified.
        
        Returns:
            A JSON object containing a natural language description of the entity.
        """
        return self._generator_chain(
            {
                "label": entity_label,
                "serialization": entity_serialization
            }
        )
    
    def tokens_used(self, str):
        return self.llm.get_num_tokens(str)
