import pandas as pd
import numpy as np

from base_model import OpenAIEmbeddingModel
from base_model import ChatModel

import openai
import numpy.linalg

openai.api_key = "YOUR-API-KEY"

class MiddleWordExtractor:
    def __init__(self, db):
        self.db = db
        self.embedmodel = OpenAIEmbeddingModel()
        self.completionmodel = ChatModel(model_type='gpt-4', )

    def word_embedding(self, user_query):
        return self.embedmodel.get_vector(user_query)

    def cosine_extractor(self, vec1, vec2):
        return np.dot(vec1, vec2) / (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2))

    def gettenpro(self, user_query):
        user_query_vector = self.word_embedding(user_query)
        cosinelist = []
        for i in range(len(self.db)):
            cosinelist.append(self.cosine_extractor(self.db['embeddings'].iloc[i], user_query_vector))
        self.db['cosinelist'] = cosinelist
        return self.db.sort_values('cosinelist', ascending=False)[['id', 'keyword', 'title', 'summary']][0:int(len(self.db)*0.2)].to_dict()

    def gpt_compare(self, user_query, json_file):
        message = [{'role': 'system', 'content': """Now you should result only one best choices considering query and json.
            The jsons are showing about recent IT tech issues, and about informations of that.
            You should output only id which is mostly suit with user query considering whole json files.
            Just output ONLY ONE NUMBER, integer.
                    """
            }, {'role': 'user', 'content': 'Query: ' + user_query + 'Json: ' + str(json_file)}]
        
        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=message,
        )

        return completion.choices[0].message.content

    def forward(self, user_query):
        id = self.gpt_compare(user_query, self.gettenpro(user_query))
        return id, self.db['keyword'][self.db['id'] == int(id)].tolist()[0]