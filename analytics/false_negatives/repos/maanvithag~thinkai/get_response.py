# import libraries
import os
from dotenv import load_dotenv
import constants as cts
from get_docs import GetDocuments as get_docs
from get_nearest_links import GetNearestLinks
import openai

class GetResponse:
    def __init__(self, query: str=None):
        self.query = query
        self.prompt_text = None
        # get env variables
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        # get openai gpt model to use
        self.gpt_model = cts.OPENAI_GPT_MODEL
        
    def get_response(self) -> str:
        # get prompt
        prompt_template = self.get_prompt()
                
        # get response from openai
        answer = self.get_openai_response(prompt_template)
        
        return answer
    
    def get_prompt(self) -> str:
        
        # get top 3 results for query        
        gnl = GetNearestLinks(self.query)
        top_links = gnl.get_links()
                
        # get docs from summaries json
        top_docs = get_docs(top_links).get_documents()
        # concatenate all summaries
        self.prompt_text = '\n'.join([doc['summary'] for doc in top_docs])
        
        # form prompt for openai
        prompt_template = f"""Use the below extract from articles on Philosophy to provide a summary in simple terms. Mould your summary to answer the subsequent question. 
        
        Start your response with "According to articles published by Stanford Encyclopedia of Philosphy". 
        
        If a summary cannot be provided, write "I don't know."

        Extract:
        \"\"\"
        {self.prompt_text}
        \"\"\"
        Question: {self.query}"""

        return prompt_template
            
    def get_openai_response(self, prompt_template: str) -> str:
        # set api key
        openai.api_key = self.openai_api_key
        # call openai
        openai_response = openai.ChatCompletion.create(
            messages=[
                {'role': 'system', 'content': 'You can summarize texts on Philosophy.'},
                {'role': 'user', 'content': prompt_template},
            ],
            model=self.gpt_model,
            temperature=0,
        )
        answer = openai_response['choices'][0]['message']['content']        
        return answer
    
if __name__ == "__main__":
    user_query = input("Hello, Welcome to ThinkAI. This is Nomí, ask me a question: ")
    gr = GetResponse(user_query)
    print(gr.get_response())