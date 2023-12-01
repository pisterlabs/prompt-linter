import os
from time import sleep, time
from typing import Optional

import datasets
from tools import methods
from nltk.tokenize import word_tokenize
import openai


openai.organization = "XXX"
openai.api_key = "XXX"
MODEL = "gpt-3.5-turbo-0613"

data_name= "ecthr_a"
path = "result_original_"+data_name+"_chatgpt.pkl"

dataset_name_question = "ECTHR"
articles = ["Article 2", "Article 3", "Article 5", "Article 6", "Article 8", "Article 9", "Article 10", "Article 11", "Article 14", "Article 1 of Protocol 1"]

pick = methods.getFromPickle("../generative_ai/ecthr_a_selected_bias.pkl", "rb")


errors_list = pick['body']+pick['race']+pick['gender']

if os.path.isfile(path):
    result =  methods.getFromPickle(path, "rb")
else:  result  = []
for case in range(len(result),len(errors_list)):
    line = errors_list[case]
    truncated_case = line[3][5]
    print(">>>Doing case ", case, "/", len(errors_list))
    article_violated = []
    answers = []
    for article_num in range(0,len(articles)):
        question = "Is "+ articles[article_num] + " of the "+ dataset_name_question + " violated in the following case : " + truncated_case +"."
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": " Try to give the absolute shortest answer possible with yes or no."},
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_tokens=1,
        )
        answer = response['choices'][0]['message']['content']
        answers.append(answer)
        sleep(0.3)
        if answer.lower().replace(' ', '').startswith("yes") : article_violated.append(article_num)
    result.append([article_violated, answers]+line)
    methods.writePickle(result, path, 'wb')

print("Done")
