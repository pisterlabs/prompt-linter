import numpy as np
import os
import csv
import random
import json
import math
import jsonlines
import openai
import time
from openai.error import RateLimitError

train_folder = 'PP'
test_folder = 'PP'
write_test_folder = 'train_'+train_folder+'_test_'+test_folder

train_files = ['template', 'retrieve', 'template_retrieve', 'template_retrieve_prompt', 'template_retrieve_ngram_prompt']
test_files = ['template_retrieve', 'template_retrieve', 'template_retrieve', 'template_retrieve_prompt', 'template_retrieve_ngram_prompt']

# Replace these with the corresponding model ids
ft_models = ['MODEL_ID_template',
            'MODEL_ID_retrieve',
            'MODEL_ID_template_retrieve',
            'MODEL_ID_template_retrieve_prompt',
            'MODEL_ID_template_retrive_ngram_prompt']

for i in range(len(train_files)):

    file = train_files[i]
    test_file = test_files[i]
    ft_model = ft_models[i]

    print("Started predicting on:", file)

    f_csv = open('./output/'+ train_folder + '/'+ write_test_folder +'/' + file + '.csv', 'a', encoding = 'utf-8')
    writer = csv.writer(f_csv)
    writer.writerow(['Context', 'Ground Truth', 'Prediction'])

    count = 0

    with jsonlines.open('training_data/' + test_folder + '/' + test_file + '/' + 'advise_test.jsonl') as reader:
        for obj in reader:
            prompt_ = obj['prompt']
            complete_ = obj['completion']

            exception_occured = True

            while exception_occured:
              try:
                res = openai.Completion.create(model=ft_model, prompt=prompt_)
                exception_occured = False
              except RateLimitError:
                print("Rate Limit Error !!!")
                time.sleep(4)

            predicted = res["choices"][0]['text']
            new_row = [prompt_, complete_, predicted]
            print(new_row)
            writer.writerow(new_row)
            
            time.sleep(2)

            count += 1
            if count == 200:
              break


    print("Writing completed!", file)



