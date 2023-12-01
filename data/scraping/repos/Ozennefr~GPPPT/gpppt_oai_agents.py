import os

import pandas as pd
import numpy as np
import openai

from gpppt_base_agents import Agent

openai.api_key = os.environ.get('OPENAI_KEY')

class ChatAgent(Agent):
    def setup(self):
        self.memory = self.options.get('memory', 1)
        self.model = self.options['model']
        self.messages = []

    def prompt(self, prompt): 
        self.messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages[-self.memory:],
            max_tokens=1024, n=1, stop=None, temperature=0.5,
        )
        answer = response["choices"][0]["message"]
        self.messages.append(answer)
        return answer["content"]

class VectorSearchAgent(Agent):
    def setup(self):
        self.emb_col = self.options['vector_col']
        self.txt_col = self.options['text_col']
        self.model = self.options['model']
        self.db = pd.read_feather(self.options['filename']).assign(
            **{self.emb_col: lambda df: df[self.emb_col].apply(np.array)}
        )

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(
            input = [text], model=self.model
        )['data'][0]['embedding']

    def prompt(self, prompt):
        chatbot_vector = self.get_embedding(prompt)
        similarities = [\
            np.dot(chatbot_vector, vec) 
            for vec in self.db[self.emb_col].to_list()
        ]
        most_similar_index = np.argmax(similarities)
        return self.db.iloc[[most_similar_index],:].to_dict('records')[0][self.txt_col]

agents_constructors = {
    'chat': ChatAgent, 
    'vector_search': VectorSearchAgent,
}