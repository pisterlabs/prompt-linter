import openai
import saytex
import ivy
import sklearn
import numpy as np

# define default architecture

class MyModel(ivy.Module):
    def __init__(self):
        self.linear0 = ivy.Linear(3, 64)
        self.linear1 = ivy.Linear(64, 1)
        ivy.Module.__init__(self)

    def _forward(self, x):
        x = ivy.relu(self.linear0(x))
        return ivy.sigmoid(self.linear1(x))
    
    def fit(self, x, y):
        # self._fit(x, y)
        pass

    # takes the column name and generates the embedding of a column and returns a prediction
    def predict(self, column_name):
        embedding = self.shape_column(column_name)
        return self._forward(embedding)

    def shape_column(self, column_name):
        # make gpt 3 call
        embedding = column_name
        return embedding
    
class BaseClassifier():
    def __init__(self, model):
        self.model = model

    def train(self, descriptions, labels):
        embs = [self.get_embedding(desc) for desc in descriptions]
        self.model.fit(embs, labels)
    
    def classify(self, description):
        embedding = self.get_embedding(description)
        # prediction = model.predict([embedding])[0]
        return self.model.predict([embedding])[0]
    
    def classify_list(self, descriptions):
        return [
            (description, self.classify(description))
            for description in descriptions
        ]
    
    def get_embedding(self, text, model="text-similarity-ada-001"):
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

class ColumnClassifier(BaseClassifier):
    def __init__(self, model):
        super().__init__(model)
    
class FuncClassifier(BaseClassifier):
    def __init__(self, model):
        super().__init__(model)
