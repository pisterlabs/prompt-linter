from typing import Tuple, Any
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
import torch

# for not seeing a warning message
import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from keras.models import model_from_json

import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

@tf.keras.utils.register_keras_serializable()
class WeightedCosineSimilarity(tf.keras.layers.Layer):

    def __init__(self, units = 128, activation=None, **kwargs):
        '''Initializes the class and sets up the internal variables'''

        super(WeightedCosineSimilarity, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def get_config(self):
        config = super(WeightedCosineSimilarity, self).get_config()
        return config


    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # W should be half the size of the input and should be ones
        w_init = tf.ones_initializer()
        w_init_val = w_init(shape=(int(input_shape[-1] / 2),), dtype='float32')
        self.w = tf.Variable(initial_value=w_init_val, trainable='true')



    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        # Take the first half of the input which is U:
        U = inputs[:, :int(inputs.shape[-1] / 2)] # (128, 768)
        # Take the second half of the input which is V:
        V = inputs[:, int(inputs.shape[-1] / 2):] # (128, 768)

        # Compute the element wise product of U, V and W
        UW = tf.multiply(U, tf.exp(self.w)) # (128, 768) * (768)
        # Compute the multiplication of UW and V

        VW = tf.multiply(V, tf.exp(self.w)) # (128, 768) * (768)

        UWVW = tf.multiply(UW, VW) # (128, 768) * (768)

        # Sum the result over the second axis
        WUV = tf.reduce_sum(UWVW, axis=1) # (128, 768) -> (128, 1)
        # Square UW and VW
        WU_squared = tf.square(UW) # (128, 768) -> (128, 768)
        WV_squared = tf.square(VW) # (128, 768) -> (128, 768)

        # Sum the result over the second axis
        WU_squared_sum = tf.reduce_sum(WU_squared, axis=1)  # (128, 768) -> (128, 1)
        WV_squared_sum = tf.reduce_sum(WV_squared, axis=1)  # (128, 768) -> (128, 1)

        # take the root of the sum of squares of WUV, WU_squared and WV_squared
        WU_squared_root = tf.sqrt(WU_squared_sum) # (128, 1)
        WV_squared_root = tf.sqrt(WV_squared_sum) # (128, 1)

        denominator = tf.multiply(WU_squared_root, WV_squared_root) # (128, 1) * (128, 1) = (128, 1)

        # divide WUV by the denominator
        WUV_div_denominator = tf.divide(WUV, denominator)

        return self.activation(WUV_div_denominator)


def pipeline(query: str, method: str = 'cs', n_contexts: int = 5, chatgpt_prompt = "You are a Teachers Assistant and you should answer the QUESTION using the information given in the CONTEXT, if the CONTEXT is unrelated, you should ignore it."):
    """
    This function is the pipeline for the entire project. It takes in a query and finds the most relevant document.
    and gives it to the OpenAI API to generate a answer
    :param n_contexts: The number of contexts to return
    :param semantic_search_model: The semantic search model to use
    :param query: The query to search for
    :param n_contexts: The number of contexts to return
    :param method: The semantic search model to use

    :return:
    """
    # 1. Preprocess the query
    embedding = get_text_embedding(query)
    # 2. Semantic Search
    best_ctx = semantic_search_model(embedding, method, n_contexts)
    # 3. Answer Generation
    answer = answer_generation(query, best_ctx, chatgpt_prompt=chatgpt_prompt)
    # 4. Return the answer
    return answer, best_ctx


def get_text_embedding(text: str):
    # Load pre-trained model and tokenizer
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
    embedding = model.encode(text, convert_to_tensor=True)

    return embedding


def semantic_search_model(embedding: np.ndarray, method: str = 'ann', n_contexts: int = 5) -> str:
    """
    This function takes in a query embedding and finds the most relevant document by using the ANN
    :param n_contexts:  The number of contexts to return
    :param embedding: The query embedding of the query of dimension 768
    :return: The most relevant n contexts
    """
    # load the dataset
    df = pd.read_pickle('./Data_Generation/df_pickle/final_02450_emb.pkl')
    # load the embeddings
    # make the df only contain the unique contexts
    df = df.drop_duplicates(subset=['context'])
    context_embeddings = np.stack(df['context_embedding'].to_numpy())
    # Concatonate the query embedding ontop of each element in the context_embeddings, so each row is the query embedding and the context embedding
    model_input = np.concatenate((np.tile(embedding, (len(context_embeddings), 1)), context_embeddings),
                                 axis=1)

    if method == 'ann':
        with open('ANN/ANN_resamp.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("ANN/ANN_resamp.h5")
        print("Loaded model from disk")

        # Predict the most relevant context
        prediction = loaded_model.predict(model_input)
        # Get the 2 most relevant contexts
        index = np.argsort(prediction, axis=0)[-n_contexts:]
        # Get the context
       # best_ctx_lst = [df.iloc[i]['context'].to_numpy()[0] for i in index]
       # best_ctx = '. '.join(best_ctx_lst)
        best_ctx_lst = [df.iloc[i]["context"].to_numpy()[0] for i in index]
        best_ctx = '\n'.join(best_ctx_lst)

        return best_ctx


    if method == 'weighted_cs':
        with open('ANN/model_cos.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, custom_objects={'CustomLayer': WeightedCosineSimilarity})
        # load weights into new model
        loaded_model.load_weights("ANN/model_cos.h5")
        print("Loaded model from disk")
        # Predict the most relevant context
        prediction = loaded_model.predict(model_input)
        # Get the 2 most relevant contexts
        index = np.argsort(prediction, axis=0)[-n_contexts:]
        # Get the context
        best_ctx_lst = [df.iloc[i]['context'].to_numpy()[0] for i in index]
        best_ctx = '\n'.join(best_ctx_lst)

        return best_ctx



    elif method == 'cs':
        # Should simply be the cosine similarity
        x = np.array(embedding)
        y = context_embeddings
        cos_sim = np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y, axis=1))
        # Get the 2 most relevant contexts
        index = np.argsort(cos_sim, axis=0)[-n_contexts:]
        # Get the context
        #best_ctx_lst = [df.iloc[i]['context'] for i in index]
        #best_ctx = '. '.join(best_ctx_lst)

        best_ctx_lst = [df.iloc[i]["context"] for i in index]
        best_ctx = '\n'.join(best_ctx_lst)

        return best_ctx


    return "ERROR: Possible methods are 'ann', 'weighted_cs' and 'cs'"


def answer_generation(query: str, context: str = "", pipeline_mode=True, chatgpt_prompt = "You are a Teachers Assistant and you should answer the QUESTION using the information given in the CONTEXT, if the CONTEXT is unrelated, you should ignore it."):

    """
    This function takes in a query and a context and uses the OpenAI API to generate an answer
    :param query:
    :param context:
    :return:
    """
    if pipeline_mode:
        print(f'CONTEXT: ```{context}``` QUESTION: ```{query}``` ANSWER:')
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,

                #messages=[
                 #   {"role": "system",
                 #    "content": chatgpt_prompt},
                 #   {"role": "user", "content": f'CONTEXT: ```{context}``` QUESTION: ```{query}``` ANSWER:'},
                #]

            messages = [
                {"role": "user", "content": f'  {chatgpt_prompt},CONTEXT: ```{context}``` QUESTION: ```{query}``` ANSWER:'},
            ]
            )
        except Exception as e:
            return "OPENAI_ERROR:", str(e)


    else:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {"role": "user", "content": f'QUESTION: ```{query}``` ANSWER:'},
                    ]

            )
        except Exception as e:
            return "OPENAI_ERROR:", str(e)

    return completion.choices[0].message.content


if __name__ == "__main__":
    # Read in the data
    query = "What is the purpose of the Introduction to Machine Learning and Data Mining Lecture notes?"
    answer_pipeline = pipeline(query, method='cs', n_contexts=2)
    answer_chatgpt = answer_generation(query, pipeline_mode=False)
    print(answer_pipeline)
    print(answer_chatgpt)
