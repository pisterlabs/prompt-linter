from langchain.chat_models import ChatOpenAI
from chains.output_format.base import chain_output_format
from chains.code_generator.base import chain_code_generator
import os

class GPTAssistant():
    def __init__(self,api_key:str):
        os.environ['OPENAI_API_KEY'] = api_key
        self.llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k', request_timeout=120, client=None)

    def chain_response_format(self, html_content):
        # prompt templates
        output_format_chain = chain_output_format(self.llm)

        # chain
        return output_format_chain.run(html_content=html_content)
    
    def chain_code_generator(self, output_format, html_content):
        # Prompt templates
        script_chain = chain_code_generator(self.llm)

        return script_chain.run(output_format=output_format, html_content=html_content)
