# Import required libraries
import math 
import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import pandas as pd
import numpy as np
import time

# tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
# model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

answers = pd.read_csv('subtaskA_answers_all.csv', header=None, delimiter = ',')
data = pd.read_csv('subtaskA_data_all.csv', header=0, delimiter = ',')
data = np.asarray(data)
answers = np.array(answers)

for x in range(len(data)): # lower all
    for y in range(0,2):
        temp = str(data[x][y])
        data[x][y] = temp.lower()

def score(sentence):
    tensor_input = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    outputs = model(tensor_input, labels=tensor_input)
    loss, logits = outputs[:2]
    return math.exp(loss)


loaded_data = np.load('all_data.npy')
wrong_preds = pd.read_csv('wrong_preds.csv', header=0, delimiter='--', engine='python')







""" *** *** *** *** *** FOR THE LOVE OF GOD DON'T DELETE ANYTHING BELOW THIS *** *** *** *** *** """

# seconds = time.time()
# start = time.ctime(seconds)

# all_data = []
# for x in range(len(data)):
#     temp0 = score(data[x][1])
#     temp1 = score(data[x][2])
#     # take option with higher perplexity score (AGAINST common sense)
#     if temp0 < temp1: #[index, sent0, sent0_score, sent1, sent1_score, answer, our_pred]
#         all_data.append(np.array([data[x][0], data[x][1], temp0, data[x][2], temp1, answers[x][1], 1]))
#     elif temp0 > temp1:
#         all_data.append(np.array([data[x][0], data[x][1], temp0, data[x][2], temp1, answers[x][1], 0]))
#     else: # same score (same sentence), add sent0 as pred idx
#         all_data.append(np.array([data[x][0], data[x][1], temp0, data[x][2], temp1, answers[x][1], 0])) 

# all_data = np.array(all_data)
# np.save('all_data.npy', all_data) # save array for later use

# end = time.time()
# print('Start: ', start)
# print('End! ', time.ctime(end))
# print(end - seconds)




# np.savetxt('sents_scores_preds.csv', loaded_data, delimiter='--', fmt="%s", encoding='utf-8',
#             header='id,sent0,sent0_score,sent1,sent1_score,answer,prediction')

# wrong_preds = [] # change for new array format
# #wrong_preds.append(['id', 'sent0', 'sent0_score', 'sent1', 'sent1_score', 'answer', 'prediction'])
# for x in range(len(loaded_data)):
#     if int(loaded_data[x][5]) != int(loaded_data[x][6]):
#         wrong_preds.append(loaded_data[x])
# wrong_preds_np = np.array(wrong_preds)
# np.savetxt('wrong_preds.csv', wrong_preds_np, delimiter='--', fmt="%s", encoding='utf-8', 
#             header='id,sent0,sent0_score,sent1,sent1_score,answer,prediction')
