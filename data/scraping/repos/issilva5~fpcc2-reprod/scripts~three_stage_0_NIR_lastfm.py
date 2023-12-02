# !pip install openai

import time
import numpy as np
import json
import openai
import random 
import argparse
import logging
import datetime

logging.basicConfig(filename='log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

parser = argparse.ArgumentParser()

parser.add_argument('--length_limit', type=int, default=8, help='')
parser.add_argument('--num_cand', type=int, default=19, help='')
parser.add_argument('--random_seed', type=int, default=2023, help='')
parser.add_argument('--api_key', type=str, default="sk-", help="")

args = parser.parse_args()

rseed = args.random_seed
random.seed(rseed)

def read_json(file):
    with open(file) as f:
        return json.load(f)

def write_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

data_ml_100k = read_json("./lastfm.json")

# print (data_ml_100k[0][0])
# print (data_ml_100k[0][1])
# print (len(data_ml_100k))


open_ai_keys = ['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']
open_ai_keys_index = 0
openai.api_key = open_ai_keys[open_ai_keys_index]


u_item_dict = {}
u_item_p = 0
for elem in data_ml_100k:
    seq_list = elem[0].split(' | ')
    for song in seq_list:
        if song not in u_item_dict:
            u_item_dict[song] = u_item_p
            u_item_p +=1
# print (len(u_item_dict))
u_item_len = len(u_item_dict)

user_list = []
for i, elem in  enumerate(data_ml_100k):
    item_hot_list = [0 for ii in range(u_item_len)]
    seq_list = elem[0].split(' | ')
    for song in seq_list:
        item_pos = u_item_dict[song]
        item_hot_list[item_pos] = 1
    user_list.append(item_hot_list)
user_matrix = np.array(user_list)
user_matrix_sim = np.dot(user_matrix, user_matrix.transpose())


pop_dict = {}
for elem in data_ml_100k:
    # elem = data_ml_100k[i]
    seq_list = elem[0].split(' | ')
    for song in seq_list:
        if song not in pop_dict:
              pop_dict[song] = 0
        pop_dict[song] += 1
        
        
        
i_item_dict = {}
i_item_id_list = []
i_item_user_dict = {}
i_item_p = 0
for i, elem in  enumerate(data_ml_100k):
    seq_list = elem[0].split(' | ')
    for song in seq_list:
        if song not in i_item_user_dict:
            item_hot_list = [0. for ii in range(len(data_ml_100k))]
            i_item_user_dict[song] = item_hot_list
            i_item_dict[song] = i_item_p
            i_item_id_list.append(song)
            i_item_p+=1
#         item_pos = item_dict[song]
        i_item_user_dict[song][i] += 1
#     user_list.append(item_hot_list)
i_item_s_list = []
for item in i_item_id_list:
    i_item_s_list.append(i_item_user_dict[item])
#     print (sum(item_user_dict[item]))
item_matrix = np.array(i_item_s_list)
item_matrix_sim = np.dot(item_matrix, item_matrix.transpose())

id_list =list(range(0,len(data_ml_100k)))



### user filtering
def sort_uf_items(target_seq, us, num_u, num_i):

    candidate_songs_dict = {} 
    sorted_us = sorted(list(enumerate(us)), key=lambda x: x[-1], reverse=True)[:num_u]
    dvd = sum([e[-1] for e in sorted_us])
    for us_i, us_v in sorted_us:
        us_w = us_v * 1.0/dvd
#         print (us_i)
        us_elem = data_ml_100k[us_i]
#         print (us_elem[0])
#         assert 1==0
        us_seq_list = us_elem[0].split(' | ')#+[us_elem[1]]

        for us_m in us_seq_list:
#             print (f"{us_m} not in {target_seq}, {us_m not in target_seq}")
#             break

            if us_m not in target_seq:
                if us_m not in candidate_songs_dict:
                    candidate_songs_dict[us_m] = 0.
                candidate_songs_dict[us_m]+=us_w
                
#         assert 1==0
                
    candidate_pairs = list(sorted(candidate_songs_dict.items(), key=lambda x:x[-1], reverse=True))
#     print (candidate_pairs)
    candidate_items = [e[0] for e in candidate_pairs][:num_i]
    return candidate_items


### item filtering
def soft_if_items(target_seq, num_i, total_i, item_matrix_sim, item_dict):
    candidate_songs_dict = {} 
    for song in target_seq:
#         print('ttt:',song)
        sorted_is = sorted(list(enumerate(item_matrix_sim[item_dict[song]])), key=lambda x: x[-1], reverse=True)[:num_i]
        for is_i, is_v in sorted_is:
            s_item = i_item_id_list[is_i]
            
            if s_item not in target_seq:
                if s_item not in candidate_songs_dict:
                    candidate_songs_dict[s_item] = 0.
                candidate_songs_dict[s_item] += is_v
#             print (item_id_list[is_i], candidate_songs_dict)
    candidate_pairs = list(sorted(candidate_songs_dict.items(), key=lambda x:x[-1], reverse=True))
#     print (candidate_pairs)
    candidate_items = [e[0] for e in candidate_pairs][:total_i]
#     print (candidate_items)
    return candidate_items



'''
In order to economize, our initial step is to identify user sequences that exhibit a high probability of obtaining accurate predictions from GPT-3.5 based on their respective candidates. 
Subsequently, we proceed to utilize the GPT-3.5 API to generate predictions for these promising user sequences.

Using this only 5 user were selected, so I removed this step.
However, I added the GT song in the candidate set, so that it becames a possibility of choice.
'''
length_limit = args.length_limit
num_u= 12
total_i = args.num_cand

temp_1 = """
Candidate Set (candidate songs): {}.
The songs I have listened (listened songs): {}.
Step 1: What features are most important to me when selecting songs (Summarize my preferences briefly)? 
Answer: 
"""

temp_2 = """
Candidate Set (candidate songs): {}.
The songs I have listened (listened songs): {}.
Step 1: What features are most important to me when selecting songs (Summarize my preferences briefly)? 
Answer: {}.
Step 2: Selecting the most featured songs from the listened songs according to my preferences (Format: [no. a listened song.]). 
Answer: 
"""

temp_3 = """
Candidate Set (candidate songs): {}.
The songs I have listened (listened songs): {}.
Step 1: What features are most important to me when selecting songs (Summarize my preferences briefly)? 
Answer: {}.
Step 2: Selecting the most featured songs (at most 5 songs) from the listened songs according to my preferences in descending order (Format: [no. a listened song.]). 
Answer: {}.
Step 3: Can you recommend 10 songs from the Candidate Set similar to the selected songs I've listened (Format: [no. a listened song - a candidate song])?.
Answer: 
"""

count = 0
total = 0
results_data = []

try:

    for i in id_list:#[:10] + cand_ids[49:57] + cand_ids[75:81]:
        logging.info(f'Running for {i}/{len(id_list)}')
        elem = data_ml_100k[i]
        seq_list = elem[0].split(' | ')[::-1]
        
        candidate_items = sort_uf_items(seq_list, user_matrix_sim[i], num_u=num_u, num_i=total_i)
        candidate_items.append(elem[1]) # TODO run with and without this
        random.shuffle(candidate_items)

        input_1 = temp_1.format(', '.join(candidate_items), ', '.join(seq_list[-length_limit:]))

        try_nums = 5
        kk_flag = 1
        while try_nums:
            try:
                response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=input_1,
                        max_tokens=512,
                        temperature=0,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n = 1,
                    )
                try_nums = 0
                kk_flag = 1
            except Exception as e:
                if 'exceeded your current quota' in str(e):

                    # open_ai_keys_index +=1
                    openai.api_key = open_ai_keys[open_ai_keys_index]
                logging.info('Going to sleep')
                print('Going to sleep')
                time.sleep(1) 
                try_nums = try_nums-1
                kk_flag = 0

        if kk_flag == 0:
            time.sleep(5) 
            response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=input_1,
                        max_tokens=256,
                        temperature=0,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n = 1,
                    )

        predictions_1 = response["choices"][0]['text']
        
        
        input_2 = temp_2.format(', '.join(candidate_items), ', '.join(seq_list[-length_limit:]), predictions_1)

        try_nums = 5
        kk_flag = 1
        while try_nums:
            try:
                response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=input_2,
                        max_tokens=512,
                        temperature=0,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n = 1,
                    )
                try_nums = 0
                kk_flag = 1
            except Exception as e:
                if 'exceeded your current quota' in str(e):

                    # open_ai_keys_index +=1
                    openai.api_key = open_ai_keys[open_ai_keys_index]
                logging.info('Going to sleep')
                print('Going to sleep')
                time.sleep(1) 
                try_nums = try_nums-1
                kk_flag = 0

        if kk_flag == 0:
            time.sleep(5) 
            response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=input_2,
                        max_tokens=256,
                        temperature=0,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n = 1,
                    )

        predictions_2 = response["choices"][0]['text']
        
        
        input_3 = temp_3.format(', '.join(candidate_items), ', '.join(seq_list[-length_limit:]), predictions_1, predictions_2)

        try_nums = 5
        kk_flag = 1
        while try_nums:
            try:
                response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=input_3,
                        max_tokens=512,
                        temperature=0,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n = 1,
                    )
                try_nums = 0
                kk_flag = 1
            except Exception as e:
                if 'exceeded your current quota' in str(e):

                    # open_ai_keys_index +=1
                    openai.api_key = open_ai_keys[open_ai_keys_index]
                logging.info('Going to sleep')
                print('Going to sleep')
                time.sleep(1) 
                try_nums = try_nums-1
                kk_flag = 0

        if kk_flag == 0:
            time.sleep(5) 
            response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=input_3,
                        max_tokens=256,
                        temperature=0,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n = 1,
                    )

        predictions = response["choices"][0]['text']
        

        hit_=0
        if elem[1] in predictions:
            count += 1
            hit_ = 1
        else:
            pass
        total +=1
        
        
        
        # print (f"input_1:{input_1}")
        # print (f"predictions_1:{predictions_1}\n")
        # print (f"input_2:{input_2}")
        # print (f"predictions_2:{predictions_2}\n")
        # print (f"input_3:{input_3}")
        # print (f"GT:{elem[1]}")
        # print (f"predictions:{predictions}")
        
        # print (f"GT:{elem[-1]}")
        print (f'PID:{i}; Hit@10:{count}/{total}={count*1.0/total}\n')
        logging.info(f'PID:{i}; Hit@10:{count}/{total}={count*1.0/total}\n')
        result_json = {"PID": i,
                    "Input_1": input_1,
                    "Input_2": input_2,
                    "Input_3": input_3,
                    "GT": elem[1],
                    "Predictions_1": predictions_1,
                    "Predictions_2": predictions_2,
                    "Predictions": predictions,
                    'Hit': hit_,
                    'Count': count,
                    'Current_total':total,
                    'Hit@10':count*1.0/total}
        results_data.append(result_json)

finally:
    
    file_dir = f"./{str(datetime.datetime.now())}-lastfm_results_multi_prompting_len{length_limit}_numcand_{total_i}_seed{rseed}.json"
    write_json(results_data, file_dir)
    
    

    
    
