from transformers import HfAgent
import guidance
import os

class StarCoderAgent:
    def __init__(self):
        self._agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
        
    def run(self, query: str) -> str:
        " Useful for answering questions about code "
        print("Got to the CodeAgent tool")
        return self._agent.run(query, temperature=0.7, remote=True)
    
class AzureCodeAgentExplain:
    query_template = "Please give your answer to the following question. Pay particular attention to any possible security issues:\n{{query}}"
    def __init__(self):
        self.guide = guidance.llms.OpenAI(
            'code-davinci-002',
            api_type='azure',
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_base=os.environ.get('OPENAI_API_BASE'),
            api_version=os.environ.get('OPENAI_API_VERSION'),
            deployment_id=os.environ.get('OPENAI_DEPLOYMENT_NAME'),
            caching=False
        )
        
    def run(self, query: str) -> str:
        " Useful for answering questions about code "
        return guidance(self.query_template, query=query, llm=self.guide)()
    