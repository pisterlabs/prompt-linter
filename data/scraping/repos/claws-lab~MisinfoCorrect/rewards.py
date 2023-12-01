from .automated_metrics import *
import numpy as np 
import nltk
# input: edited sentence, initial sentence, weights
# from .empathy_classifier_bi_encoder_attention import empathy_classifier
from .coherence_classifier2 import coherence_classifier

from .coherence_classifier2 import politeness_classifier
from .coherence_classifier2 import refutation_classifier
from .coherence_classifier2 import evidence_classifier

orig_sent_list = ["It might help to re-install Python if possible","The version might behind the bug."]
new_sent_list = ["The version might be the reason for the bug.","The version might be the reason behind the bug."]

# w: weight for the importance of the rewards
w = {'edit':1,'bleu':1,'dist1':10,'dist2':1,'pp':10,'spec':1,'empathy':1, 
     'politeness':1, 'refutation':1, 'evidence':1, 'coherence':0.1, 'relevance':0.1}

def calc_rewards(seeker_posts, original_responses, generated_responses, candidates = None,
				_edit=False,
				_bleu=False,
				_distinct=False,
				_perplexity=False,
				_specificity=False,
				_empathy=False,
				_empathy_change=False,
				_coherence=False,
				_add_noise=True,
				_pick_categorical='',
				_empathy_adaptive=False,
				_politeness=False,
				_refutation=False,
				_evidence=False,
				_relevance=False,
				NOISE=0.00001):

	total_score = 0

	if _edit:
		edit = edit_level_jaccard(original_responses, generated_responses)
		total_score += edit*w['edit']
	
	if _bleu:
		bleu_score = bleu(generated_responses, original_responses)
		total_score += bleu_score*w['bleu']
	
	if _distinct:
		distinct_1, distinct_2 = distinct(generated_responses)
		total_score += distinct_1*w['dist1']+distinct_2*w['dist2']
	
	if _perplexity:
		perplexity_score = perplexity(generated_responses)
		total_score += perplexity_score*w['pp']
	
	if _specificity:
		specificity_score = 0 # specificity(seeker_posts, generated_responses)
		total_score += specificity_score*w['spec']
	
	if _empathy:
		empathy_score = calc_empathy_score(seeker_posts, generated_responses)
		total_score += empathy_score*w['empathy']
	
	if _empathy_change:
		prev_empathy_score = calc_empathy_score(seeker_posts, original_responses)
		curr_empathy_score = calc_empathy_score(seeker_posts, generated_responses)

		total_score += curr_empathy_score - prev_empathy_score
	

	if _coherence:
		reward_val = relevance(seeker_posts, generated_responses)
		total_score += reward_val*w['coherence']
	if _politeness:
		reward_val = calc_politeness_score(generated_responses, "")
		total_score += reward_val*w['politeness']
	if _refutation:
		reward_val = calc_refutation_score(seeker_posts, generated_responses)
		total_score += reward_val*w['refutation']
	if _evidence:
		reward_val = calc_evidence_score(seeker_posts, generated_responses)
		total_score += reward_val*w['evidence']
	if _relevance:
		reward_val = relevance(seeker_posts, generated_responses)
		total_score += reward_val*w['relevance']


	if _add_noise:
		total_score -= NOISE
	
	if _empathy_adaptive:
		_,_,_,_,ER_score, IP_score, EX_score = calc_empathy_score_3dim(seeker_posts, generated_responses)
		total_score += ((2-ER_score)*ER_score+(2-IP_score)*IP_score+(2-EX_score)*EX_score)*w['empathy']*0.5

	return total_score


def edit_level_jaccard(orig_sent_list, new_sent_list):
	total_score = 0
	for i, orig_sent in enumerate(orig_sent_list):
		total_score += (nltk.jaccard_distance(set(orig_sent), set(new_sent_list[i])))
	return total_score/len(orig_sent_list)

edit_level_jaccard(orig_sent_list, new_sent_list)

def calc_empathy_score(seeker_posts, generated_responses):
	batch_score = 0
	for i in range(len(seeker_posts)):
		(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX) = empathy_classifier.predict_empathy([seeker_posts[i]], [generated_responses[i]])
		batch_score += ((predictions_ER[0]+predictions_IP[0]+predictions_EX[0])*0.5) 
	
	return batch_score/len(seeker_posts)

def calc_empathy_score_3dim(seeker_posts, generated_responses):
	batch_score = 0
	ER_score_list =[]
	IP_score_list =[]
	EX_score_list =[]

	for i in range(len(seeker_posts)):
		try:
			(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX) = empathy_classifier.predict_empathy([seeker_posts[i]], [generated_responses[i]])
			batch_score += ((predictions_ER[0]+predictions_IP[0]+predictions_EX[0]))
			ER_score_list.append(predictions_ER[0])
			IP_score_list.append(predictions_IP[0])
			EX_score_list.append(predictions_EX[0])
		except:
			print('Error:', seeker_posts[i], generated_responses[i])
	
	ER_score = np.sum(ER_score_list)/len(seeker_posts)
	IP_score = np.sum(IP_score_list)/len(seeker_posts)
	EX_score = np.sum(EX_score_list)/len(seeker_posts)
	
	return batch_score/len(seeker_posts),ER_score_list,IP_score_list,EX_score_list, ER_score, IP_score, EX_score

def log2prob(logs):
	probs = np.divide(np.exp(logs), (1+np.exp(logs)))
	return probs

def calc_coherence_score(original_responses, candidate): # original_response: list of strings, candidate: string 
	(logits, predictions,) = coherence_classifier.predict_empathy(original_responses, candidate)
	logs_1 = [log[1] for log in logits]
	score = np.mean(log2prob(logs_1))
	return score

## for politeness, we only use the original response: 1 input
def calc_politeness_score(original_responses, candidate): # original_response: list of strings, candidate: string
	(logits, predictions,) = politeness_classifier.predict_empathy(original_responses, candidate)
	# here logits: 0:dim: impolite; 1:polite(normal and polite)
	logs_1 = [log[1] for log in logits]
	score = np.mean(log2prob(logs_1))
	return score

## for refutation scores, we should use two inputs: 
def calc_refutation_score(original_responses, candidate): # original_response: list of strings, candidate: string
	(logits, predictions,) = refutation_classifier.predict_empathy(original_responses, candidate)
	# here logits: 0:dim: impolite; 1:polite(normal and polite)
	logs_1 = [log[1] for log in logits]
	score = np.mean(log2prob(logs_1))
	return score

def calc_evidence_score(original_responses, candidate): # original_response: list of strings, candidate: string
	(logits, predictions,) = evidence_classifier.predict_empathy(original_responses, candidate)
	# here logits: 0:dim: impolite; 1:polite(normal and polite)
	logs_1 = [log[1] for log in logits]
	score = np.mean(log2prob(logs_1))
	return score





