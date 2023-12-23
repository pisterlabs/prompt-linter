import openai

prompt_template = """Use the following pieces of context to answer the question at the end.

{context}

Question: {question}

Helpful Answer:"""  


class GoogleSearch:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.client = openai.ChatCompletion
        self.search = self.init_search_engine()
        self.model_name = model_name
        self.template = prompt_template
    
    def init_search_engine(self):
        from langchain import GoogleSerperAPIWrapper
        search_engine=GoogleSerperAPIWrapper()
        
        return search_engine
        
    def run(self, input):
        content = self.search.run(input)
        prompts = self.template.format(context=content, question=input)
        completion = self.client.create(model=self.model_name, messages=[{"role": "user", "content": prompts}])
        content = completion.choices[0].message.content
            
        return content
    
    
# prompt_template = """Use the following pieces of context to answer the question at the end.:
# {search_result}

# Based on the web search content, answer the following question, if you don't know the answer, just say that you don't know, don't try to make up an answer:
# {question}

# Answer:
# """   