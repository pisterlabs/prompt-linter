import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from langchain.chat_models import ChatOpenAI
from sentence_transformers import CrossEncoder
from langchain.evaluation.qa import ContextQAEvalChain

import warnings
warnings.filterwarnings("ignore")

def llm_evalchain(context_examples: dict, 
                  predictions: dict) -> List:
    """
    Evaluation strategy based on ContextQAEvalChain
    as discussed in official documentation of Langchain
    Ref -> https://python.langchain.com/docs/guides/evaluation/question_answering

    Parameters
    ----------
    context_examples: dict
                      Question-Ideal Answer pair
    predictions: dict
                Question-Predicted Answer pair
    
    Returns
    -------
    graded_outputs: list
                    List of graded scores for each predicted answer
                    corresponding to the ideal answers

    """
    load_dotenv()

    openai_api_key = os.getenv('OPENAI_API_KEY')
    llm_name = os.getenv('LLM_NAME')

    llm = ChatOpenAI(model_name=llm_name, temperature=0)
     
    eval_chain = ContextQAEvalChain.from_llm(llm)

    graded_outputs = eval_chain.evaluate(context_examples, 
                                         predictions, 
                                         question_key="question", 
                                         prediction_key="text")

    graded_outputs = [1 if item['text'] == 'CORRECT' else 0
                      for item in graded_outputs]
    
    return graded_outputs

def semantic_similarity(ideal_ans:str, 
                        pred_ans:str, 
                        method:str='bi_encoder') -> float:
    """
    Calculates the semantic similarity score using the following
    methods:
    1. Bi-Encoder Score
    2. Semantic Similarity 

    Read more about both of these approaches here:
    https://www.sbert.net/examples/applications/cross-encoder/README.html

    Parameters
    ----------
    ideal_ans: str
               Ideal answer for given question
    pred_ans: str
              Answer predicted by the model for given question
    method: str 
            Choose from ['bi_encoder', 'sas]

    Returns
    -------
    score: float
           Semantic similarty score between the ideal and prediced
           answers
    """

    if method == 'bi_encoder':
        model_name = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    elif method == 'sas':
        model_name = "cross-encoder/stsb-roberta-large"
    else:
        raise NotImplementedError

    model = CrossEncoder(model_name, max_length=512)
    score = model.predict([[ideal_ans, pred_ans]])
    return score

if __name__ == '__main__':
    csv_path = 'final_result.csv'
    df = pd.read_csv(csv_path)
    df['SAS Score'] = [np.nan]*len(df)
    df['Similarity Score'] = [np.nan]*len(df)

    context_examples = []
    predictions = []

    for i in tqdm(range(len(df)), desc="Evaluating"):
        df.loc[i,['Similarity Score']] = semantic_similarity(df.iloc[i]['Ideal Answer'],
                                                      df.iloc[i]['Predicted Answers'],
                                                      method="bi_encoder")
        
        df.loc[i,['SAS Score']] = semantic_similarity(df.iloc[i]['Ideal Answer'],
                                                      df.iloc[i]['Predicted Answers'],
                                                      method="sas")
        
        context_examples.append({"question" :df.iloc[i]['Question'], 
                                "context":df.iloc[i]['Ideal Answer']})

        predictions.append({"question" :df.iloc[i]['Question'], 
                                "text":df.iloc[i]['Predicted Answers']})

    df['Context QA Eval'] = llm_evalchain(context_examples=context_examples,
                                        predictions=predictions)


    df.to_csv(csv_path, index=None)