#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm.notebook import tqdm
import pandas as pd
import openai
import json
import os

openai.api_key = os.environ["OPENAI_API_KEY"]


# In[2]:


wiki_sci_df = pd.read_parquet("top_20k_wiki_sci_embd_clusters.parquet").reset_index(drop=True)

# calculate word count from 'text' column
wiki_sci_df['word_count'] = wiki_sci_df['text'].apply(lambda x: len(str(x).split(" ")))
wiki_sci_df['section_title'] = wiki_sci_df['title']
wiki_sci_df['page'] = wiki_sci_df['title']


wiki_sci_df.head()


# In[3]:


wc_per_page = wiki_sci_df.groupby("page")[["word_count"]].sum().sort_values("word_count", ascending=False)
wc_per_page["token_count"] = (wc_per_page["word_count"] / 0.75).astype(int)
wc_per_page[:20]


# In[4]:


# sum all token_count
wc_per_page['token_count'].sum() / 1e8


# In[5]:


# remove rows where "List of" is in the page
wiki_sci_df = wiki_sci_df[~wiki_sci_df['page'].str.contains('List of')]


# In[6]:


wc_per_page = wiki_sci_df.groupby("page")[["word_count"]].sum().sort_values("word_count", ascending=False)
wc_per_page["token_count"] = (wc_per_page["word_count"] / 0.75).astype(int)
wc_per_page[:20]


# In[7]:


# sum all token_count
wc_per_page['token_count'].sum() / 1e8


# # Expected cost: 	
# 
# 
# GPT3.5 cost for 16k context is $0.004 / 1K tokens.
# 
# There is 1e8 tokens in the dataset, so the cost is 1e8 * 0.004 / 1e3 = $400 for the whole wiki sci dataset.
# 
# There is about 130 text examples per cluster, so the cost per cluster is $400 / 130 = $3.07

# In[8]:


(wc_per_page['token_count']  > 16e3).sum() / len(wc_per_page)


# In[9]:


# openai.Model.list()


# In[10]:


import pydantic


pydantic.__version__


# In[11]:


from typing import *
from pydantic import BaseModel
import json


class MultipleChoiceQuestion(BaseModel):
    question: str
    A: str
    B: str
    C: str
    D: str
    E: str
    answer: Literal["A", "B", "C", "D", "E"]

        
class MultipleChoiceQuestionList(BaseModel):
    questions: List[MultipleChoiceQuestion]

        
schema = MultipleChoiceQuestionList.model_json_schema()
# print(json.dumps(schema, indent=4))


# In[12]:


from pathlib import Path


# out_file_name = "raw_questions"
# out_file_name = "raw_questions_2"
out_file_name = "raw_questions_wiki_sci_4"


out_dir = Path(f"/home/viktor/Documents/kaggle/kaggle_llm/data/data_dumps/{out_file_name}")

# if the directory already exist, raise an error
out_dir.mkdir(exist_ok=True, parents=True)


# In[13]:


wiki_sci_df


# In[ ]:





# In[26]:


def generate_questions(wiki_sci_df_cluster):
    
    num_questions_per_cluster = 10000
    
    
    for cluster_question_idx  in range(num_questions_per_cluster):
        
        
        randint = np.random.randint(0, 1000000)
        out_path = f"{out_dir}/cluster-round-{cluster_question_idx}-{randint}.txt"
        
        
        # take a random page and text from wiki_sci_df_cluster
        random_row = wiki_sci_df_cluster.sample(1)
        page = random_row.page.values[0]
        text = random_row.text.values[0]
        section_title = random_row.section_title.values[0]
        
        try:
            # ==== prompt building ====
            model_name = "gpt-3.5-turbo-16k-0613"
            messages = [
                {
                    "role": "system", 
                    "content": (
                        f"You are a GOD-like AGI that is a top level expert in all of the fields in the world. "
                        + f"Your task is to design a multiple choice exam questions with 5 choices each for human experts in the field. "
                        + f"Of these 5 possible answers, one is correct and the other 4 are wrong. "
                        + f"Your exam questions will be derived from a wikipedia page: \"{page}\". "
                        + f"You'll receive a pair of section_title and text extracted from the wikipedia page. "
                        + f"Even though the answers are generated from the wikipedia page, your students are super smart, so don't go easy on them. "
                        + f"It is OK if the questions relates some concept from the given wikipedia page to another concept from another wikipedia page if you want to ask a question that is more difficult and that questions the student's understanding of both concepts. \n"
                        + f"Make sure that the exam questions (both the correct answer and the wrong answers) you design are relevant to the given section_title and text."
                        + f"The possible answers should be very similar to the exact answer. "
                        + f"Think step-by-step, and make sure that the answer is not too obvious. "
                        + f"Don't include possible answers that are like 'any of the above' or 'none of the above'. "
                        + f"Here are the section title and text: \n"
                        + f"Section title: {section_title}. Text: {text}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Generate 5 multiple choice questions, each with 5 possible answers on the given topic. You will generate it in the following way, step-by-step:\n"
                    + f"Step 1: you will generate a question and a correct answer based on the provided text. The answer is 1 or 2 sentences long, no longer than that. Write that down as the token outputs. Question should start like this: \n"
                    
                    + f"option: What is the significance of ... \n"
                    + f"option: Which of the following statements ... \n"
                    + f"option: What is ... \n"
                    + f"option: What is the difference between ... \n"
                    + f"option: What is the reason for ... \n"
                    + f"option: What is the role of ... \n"
                    + f"option: What is the definition of ... \n"
                    + f"option: What is the purpose of ... \n"
                    + f"option: What is the reason behind ... \n"
                    + f"option: What is the term used ... \n"
                    + f"option: What is the origin of ... \n"
                    + f"option: What are the two main ... \n"
                    + f"option: What is the reason that ... \n"
                    + f"option: What is the main focus ... \n"
                    + f"option: What is the function of ... \n"
                    + f"option: What is the interpretation of ... \n"
                    + f"option: What is the formalism that ... \n"
                    + f"option: What is the main advantage ... \n"
                    
                    + f"Step 2: you will generate 4 similar but wrong answers based on the correct answer that you provided in the step 1. Start the sentence the same as the answer, but only change wording in few places to make sure the answer is actually wrong. Make sure that the wrong answer is not too obviously wrong, that is as close to the correct answer as possible, not similar to other wrong answers. The number of sentences should be the same as the correct answer.\n"
                    
                    + f"Examples of question-correct answer pairs:\n"
                    + "['What is the Kutta condition?','The Kutta condition is a physical requirement that the fluid moving along the lower and upper surfaces of an airfoil meet smoothly, with no fluid moving around the trailing edge of the airfoil.'],"
                    + f"['What is the purpose of obtaining surgical resection specimens?','To remove an entire diseased area or organ for definitive surgical treatment of a disease, with pathological analysis of the specimen used to confirm the diagnosis.']"
                    + f"['Who published the first theory that was able to encompass previously separate field theories to provide a unifying theory of electromagnetism?, 'Maxwell']"
                    + f"['What is the significance of Baryon Acoustic Oscillations (BAOs) in the study of the universe?','BAOs establish a preferred length scale for baryons, which can be used to detect a subtle preference for pairs of galaxies to be separated by 147 Mpc, compared to those separated by 130-160 Mpc.']"
                }
            ]

            
            
            # ==== running model ====
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                functions=[
                    {
                        "name": "create_multiple_choice_question",
                        "description": "create a multiple choice question consisting of a question, and 5 choices: A, B, C, D, and E",
                        "parameters": schema,
                    }
                ],
                function_call={"name": "create_multiple_choice_question"},
            )

            # ==== parsing answers ====
            assistant_msg = response["choices"][0]["message"]
            response_options = assistant_msg.to_dict()["function_call"]["arguments"]
            
            
            # write response_options to out_path
            out_path_write = Path(out_path)
            out_path_write.write_text(response_options)
                
        except (
            openai.error.Timeout,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.InvalidRequestError,
            openai.error.AuthenticationError,
            openai.error.PermissionError,
            openai.error.RateLimitError,
        ) as e:
            print(f"can't do {page}: {repr(e)}, skipping for now")
            continue
    


# In[27]:


from joblib import Parallel, delayed
from tqdm import tqdm

n_jobs = 8

import numpy as np

clusters = np.array_split(wiki_sci_df, n_jobs)
# print([len(cluster) for cluster in clusters])


# In[28]:


results = Parallel(n_jobs=n_jobs)(delayed(generate_questions)(cluster) for cluster in tqdm(clusters))

