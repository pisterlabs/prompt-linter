from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from custom_llm import CustomLLM
import os
from dotenv import load_dotenv, find_dotenv
from langchain.prompts.prompt import PromptTemplate
from custom_prompt import PROMPT_TEMPLATE_LLAMA, PROMPT_TEMPLATE_OPENAI

class AnswerGeneration:
    def __init__(self, vector_store, model_name):
        self.vector_store = vector_store
        if model_name=="text-davinci-003":
            self.model = OpenAI(model=model_name, temperature=0)
            self.prompt = PROMPT_TEMPLATE_OPENAI
        elif model_name=="llama2":
            self.model = CustomLLM(api_url=os.getenv("LLAMA_LLM_URL"))
            self.prompt = PROMPT_TEMPLATE_LLAMA

    
    def _get_prompt(self, prompt_template):
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=prompt_template)
        return prompt
        
    def generate(self,
                 query,
                 candidate_docs,
                 chain_type="stuff"):
        
        prompt = self._get_prompt(self.prompt)
        chain = load_qa_chain(self.model,
                              chain_type=chain_type,
                              prompt=prompt)
        ## Generate answer
        answer = chain({"input_documents": candidate_docs,
                        "question": query},
                       return_only_outputs=True)
        

        return answer['output_text']