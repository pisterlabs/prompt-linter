"""Summary
"""
from typing import List, Any, Dict
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, SagemakerEndpoint, ConversationChain
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.schema import BaseMemory
from pydantic import BaseModel, Extra
import json

class SagemakerContentHandler(LLMContentHandler):
    """Helper class to parse Sagemaker input/output
    """
    
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        """Parse input into required format for Sagemaker
        
        Args:
            prompt (str): LLM Prompt
            model_kwargs (Dict): model tuning paramters
        
        Returns:
            bytes: Description
        """
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        """Parse sagemaker output. Return the first generated text as chatbot response
        
        Args:
            output (bytes): Bytes output from Sagemaker
        
        Returns:
            str: Chat response
        """
        response_json = json.loads(output.read().decode("utf-8"))
        print(response_json)
        return response_json['generated_texts'][0]

class LexConversationalMemory(BaseMemory, BaseModel):

    """Langchain Custom Memory class that uses Lex Conversation history
    
    Attributes:
        history (dict): Dict storing conversation history that acts as the Langchain memory
        lex_conv_context (str): LexV2 sessions API that serves as input for convo history
            Memory is loaded from here
        memory_key (str): key to for chat history Langchain memory variable - "history"
    """
    history = {}
    memory_key = "chat_history" #pass into prompt with key
    lex_conv_context = ""

    def clear(self):
        """Clear chat history
        """
        self.history = {}

    @property
    def memory_variables(self) -> List[str]:
        """Load memory variables
        
        Returns:
            List[str]: List of keys containing Langchain memory
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load memory from lex into current Langchain session memory
        
        Args:
            inputs (Dict[str, Any]): User input for current Langchain session
        
        Returns:
            Dict[str, str]: Langchain memory object
        """
        input_text = inputs[list(inputs.keys())[0]]

        ccontext = json.loads(self.lex_conv_context)
        memory = {
            self.memory_key: ccontext[self.memory_key] + input_text + "\nAI: ",
        }
        return memory

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Load memory from lex + current input into Langchain session memory
        
        Args:
            inputs (Dict[str, Any]): User input
            outputs (Dict[str, str]): Langchain response from calling LLM
        """
        input_text = inputs[list(inputs.keys())[0]]
        output_text = outputs[list(outputs.keys())[0]]

        ccontext = json.loads(self.lex_conv_context)
        self.history =  {
            self.memory_key: ccontext[self.memory_key] + input_text + f"\nAI: {output_text}",
        }


            

class SagemakerLangchainBot():

    """Create a langchain.ConversationChain using a Sagemaker endpoint as the LLM
    
    Attributes:
        chain (langchain.ConversationChain): Langchain chain that invokes the Sagemaker endpoint hosting an LLM
    """
    
    def __init__(self, prompt_template,
        sm_endpoint_name, 
        lex_conv_history="",
        region_name="" ):
        """Create a SagemakerLangchainBot client
        
        Args:
            prompt_template (str): Prompt template
            sm_endpoint_name (str): Sagemaker endpoint name
            lex_conv_history (str, optional): Lex convo history from LexV2 sessions API. Empty string for no history (first chat)
            region_name (str, optional): region where Sagemaker endpoint is deployed
        """


        prompt = PromptTemplate(
            input_variables=["chat_history", "input"],
            template=prompt_template
        )
        
        # Sagemaker endpoint for the LLM. Pass in arguments for tuning the model and
        sm_flant5_llm = SagemakerEndpoint(
            endpoint_name=sm_endpoint_name,
            region_name=region_name,
            content_handler=SagemakerContentHandler(),
            model_kwargs={"temperature":2.0,"max_length":50, "num_return_sequences":3, "top_k":50, "top_p":0.95, "do_sample":True}
        )

        # Create a conversation chain using the prompt, llm hosted in Sagemaker, and custom memory class
        self.chain = ConversationChain(
            llm=sm_flant5_llm, 
            prompt=prompt, 
            memory=LexConversationalMemory(lex_conv_context=lex_conv_history), 
            verbose=True
        )

    def call_llm(self,user_input) -> str:
        """Call the Sagemaker endpoint hosting the LLM by calling ConversationChain.predict()
        
        Args:
            user_input (str): User chat input
        
        Returns:
            str: Sagemaker response to display as chat output
        """
        output = self.chain.predict(
            input=user_input
        )
        print("call_llm - input :: "+user_input)
        print("call_llm - output :: "+output)
        return output 