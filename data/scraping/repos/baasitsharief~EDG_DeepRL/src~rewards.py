'''I/we certify that the code and data in this assignment were generated
independently, using only the tools and resources defined in the course
and that I/we did not receive any external help, coaching or contributions
during the production of this work."

References: 
1. https://github.com/lvwerra/trl/blob/master/nbs/04-gpt2-sentiment-ppo-training.ipynb
2. https://github.com/behavioral-data/Empathy-Mental-Health'''

import sys
sys.path.append("..")

import numpy as np
from scipy.spatial import distance
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from EmpathyModel.empathy_classifier import EmpathyClassifier
from .coherence_classifier import CoherenceClassifier

# coherence_model_path = input("Coherence Classifier path: ")
er_path = input("ER path: ")
ip_path = input("IP path: ")
ex_path = input("EX path: ")

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

# print("Loading Coherence Model")
# coherence_model = CoherenceClassifier(device, model_path=coherence_model_path)
# print("Coherence Model Loaded")

print(f"Loading Empathy Model on device: {device}")
empathy_model = EmpathyClassifier(device, er_path, ip_path, ex_path)
print("Empathy Model Loaded")

def edit_level_jaccard(orig_sent_list, new_sent_list):
	total_score = 0
	for i, orig_sent in enumerate(orig_sent_list):
		total_score += (nltk.jaccard_distance(set(orig_sent), set(new_sent_list[i])))
	return total_score/len(orig_sent_list)

def calc_empathy_score(seeker_posts, generated_responses):
	batch_score = 0
	for i in range(len(seeker_posts)):
		(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX,_,_,_,_,_,_) = empathy_model.predict_empathy([seeker_posts[i]], [generated_responses[i]])
		batch_score += ((predictions_ER[0]+predictions_IP[0]+predictions_EX[0])*0.5) 
	
	return batch_score/len(seeker_posts)

# def calc_coherence_score(original_responses, candidate): # original_response: list of strings, candidate: string 
# 	(logits, predictions,) = coherence_model.predict_empathy(original_responses, candidate)
# 	logs_1 = [log[1] for log in logits]
# 	score = np.mean(log2prob(logs_1))
# 	return score

# def log2prob(logs):
# 	probs = np.divide(np.exp(logs), (1+np.exp(logs)))
# 	return probs
 
def informationFlow(responses):
    r2=0
    if(len(responses) > 2):
        #2 representations obtained from the encoder for two consecutive turns pi and pi+1
        responses = tokenizer.encode_plus(
								responses,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								pad_to_max_length = True,
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
        h_pi = responses[-3]
        h_pi1 = responses[-1]
        # length of the two vector might not match
        min_length = min(len(h_pi), len(h_pi+1))
        h_pi = h_pi[:min_length]
        h_pi1 = h_pi1[:min_length]
        #cosine similarity 
        #cos_sim = 1 - distance.cosine(h_pi, h_pi1)
        cos_sim = 1 - distance.cdist(h_pi.cpu().numpy(), h_pi1.cpu().numpy(), 'cosine')
        #Handle negative cos_sim
        if np.any(cos_sim <= 0):
            r2 = - cos_sim
        else:
            r2 = - np.log(cos_sim)
        r2 = np.mean(r2)
    return r2

