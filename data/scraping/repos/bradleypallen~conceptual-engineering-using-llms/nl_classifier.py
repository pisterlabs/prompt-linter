import yaml
from langchain import HuggingFaceHub, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class NaturalLanguageClassifer:
    """Represents a natural language classfier."""
    
    def __init__(self, id, label, definition, model_name="gpt-4", temperature=0.1):
        """
        Initializes a classifier with a unique identifier, a label, and a definition.
        
        Parameters:
            id: The unique identifier for the class.
            label: The label or name of the class.
            definition: The definition of the class.
            model_name: The name of the model to be used for zero shot CoT classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.id = id
        self.label = label
        self.definition = definition
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._llm(model_name, temperature)
        self._classify_chain = self._zero_shot_chain_of_thought('./chains/classify.yaml')

    def to_json(self):
        """
        Converts the classifier to a JSON-like dictionary format.
        
        Returns:
            A dictionary with keys "id", "label", and "definition".
        """
        return {
            "id": self.id,
            "label": self.label,
            "definition": self.definition,
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
    
    def classify(self, entity_label, entity_description):
        """
        Determines whether or not an entity is in the extension of the classifier's concept.
        
        Parameters:
            entity_label: The label of the entity to be classified.
            entity_description: The description of the entity to be classified.
        
        Returns:
            A JSON object containing a classification of the entity based on the concept's definition.
        """
        return self._classify_chain(
            {
                "label": self.label, 
                "definition": self.definition, 
                "entity": entity_label,
                "description": entity_description
            }
        )
    
    def tokens_used(self, str):
        return self.llm.get_num_tokens(str)
