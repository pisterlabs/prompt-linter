import openai
import random
import numpy as np
import torch
import evaluate
import os
import json
from tqdm import tqdm
import re
import time
import codecs

class MinimalBayesRisk:

	def __init__(self, device, k_samples, batch_size=10):
		self.device = device
		self.batch_size = batch_size
		self.bertscore = evaluate.load("bertscore")
		self.k_samples = k_samples

	def compute_score(self, instance_list, score_matrix):
		predictions = [cand1 for (_, _, cand1, _) in instance_list]
		references = [cand2 for (_, _, _, cand2) in instance_list]
		score = self.bertscore.compute(predictions=predictions, references=references, lang='en', device=self.device)['f1']
		for instance_index, (index1, index2, _, _) in enumerate(instance_list):
			score_matrix[index1][index2] = score_matrix[index2][index1] = score[instance_index]

		return score_matrix

	def execute(self, candidates):
		assert len(candidates) == self.k_samples
		score_matrix = np.zeros((self.k_samples, self.k_samples))
		instances = []
		for j1, cand1 in enumerate(candidates):
			for j2, cand2 in enumerate(candidates):
				if j1 < j2:
					instances.append((j1, j2, cand1, cand2))
					if len(instances) == self.batch_size:
						score_matrix = self.compute_score(instances, score_matrix)
						instances = []

				if j1 == len(candidates) - 1 and j2 == len(candidates) - 1 and len(instances) > 0:
					score_matrix = self.compute_score(instances, score_matrix)

		sum_scores = np.sum(score_matrix, axis=1)
		final_output = candidates[np.argmax(sum_scores)]

		return final_output


class GPT3DialogueRemediator:

	def __init__(self, k_samples, gpt3_model_name='text-davinci-003'):
		self.k_samples = k_samples
		self.keys = ["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",
					 "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",
		  			 "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",
		  			 "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",
		  			 "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"]
		self.gpt3_model_name = gpt3_model_name

	def execute(self, context, response, violated_norm_type):
		context_str = ""
		for i, utterance in enumerate(context):
			if i % 2 == 0:
				context_str += "Me: %s " % utterance
			else:
				context_str += "Other: %s " % utterance
		role = "Other" if len(context) % 2 == 0 else "Me"
		response = role + ": " + response
		prompt_str = f"Here is the dialogue context: [{context_str}]. Here is the current response [{response}]. Explaining why the current response is not appropriate in terms of {violated_norm_type}."

		request_success_flag = False
		for time_count in range(3):
			try:
				response = openai.Completion.create(
					model=self.gpt3_model_name,
					api_key=random.choice(self.keys),
					prompt=prompt_str,
					temperature=0.9,
					max_tokens=128,
					n=self.k_samples
				)
			except:
				print("request GPT3 fail (%d / 3)" % (time_count + 1))
				time.sleep(5)
				continue

			request_success_flag = True
			break

		if not request_success_flag:
			return []

		remediations = []
		for choice in response["choices"]:
			remediations.append(choice['text'].strip())

		return remediations

class GPT3DialogueCorrector:

	def __init__(self, k_samples, gpt3_model_name='text-davinci-003'):
		self.k_samples = k_samples
		self.keys = ["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",
					 "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",
		  			 "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",
		  			 "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",
		  			 "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"]
		self.gpt3_model_name = gpt3_model_name
		self.correction_reasons = {
			'apology': "should make an appropriate apology",
			'persuasion': "should make an appropriate persuasion",
			'greeting': 'should make an appropriate greeting',
			'request': 'should make an appropriate request',
			'criticism': 'should make an appropriate criticism'
		}

	def execute(self, context, response, violated_norm_type):
		context_str = ""
		for i, utterance in enumerate(context):
			context_str += "%s " % utterance
		prompt_str = f"Here is the dialogue context: [{context_str}]. Here is the current response [{response}]. Here is another response which {self.correction_reasons[violated_norm_type]} ["

		request_success_flag = False
		for time_count in range(3):
			try:
				response = openai.Completion.create(
					model=self.gpt3_model_name,
					api_key=random.choice(self.keys),
					prompt=prompt_str,
					suffix=']',
					temperature=0.9,
					max_tokens=128,
					n=self.k_samples
				)
			except:
				print("request GPT3 fail (%d / 3)" % (time_count + 1))
				time.sleep(5)
				continue

			request_success_flag = True
			break

		if not request_success_flag:
			return []

		corrections = []
		for choice in response["choices"]:
			corrections.append(choice['text'].strip())

		return corrections


if __name__ == "__main__":
	mini_BR = MinimalBayesRisk('cuda', 5)
	remediator = GPT3DialogueRemediator(5)
	corrector = GPT3DialogueCorrector(5)

	context = ['A: Hello. B: Who are you?']
	response = "I just say hello!"
	corrections = corrector.execute(context, response, 'greeting')
	final_correction = mini_BR.execute(corrections)

	print(final_correction)

	# path = 'clear_recheck/'
	#
	# for filename in os.listdir(path):
	# 	if (not filename.endswith('.json')) or (not filename.startswith('id')) or os.path.exists(f'clear_recheck/gpt3_remediation_output_English_context_{filename[:-5]}.json'):
	# 		continue
	#
	# 	new_output = {'annotations': []}
	# 	print(filename)
	# 	with open(f'clear_recheck/{filename}') as out:
	# 		data = json.loads(out.read())
	#
	# 	for dialogue in tqdm(data['annotations']):
	# 		context = [c['utterance'] for c in dialogue['dialogue'][:-1]]
	# 		response = dialogue['dialogue'][-1]['utterance']
	# 		remediations = remediator.execute(context, response, dialogue['norm rule']['norm_type'])
	# 		corrections = corrector.execute(context, response, dialogue['norm rule']['norm_type'])
	#
	# 		final_correction = mini_BR.execute(corrections)
	# 		final_remediation = mini_BR.execute(remediations)
	# 		dialogue['gpt3_corrections'] = corrections
	# 		dialogue['gpt3_remediations'] = remediations
	# 		dialogue['gpt3_final_correction'] = final_correction
	# 		dialogue['gpt3_final_remediation'] = final_remediation
	#
	# 		new_output['annotations'].append(dialogue)
	#
	#
	# 	with codecs.open(f'clear_recheck/gpt3_remediation_output_English_context_{filename[:-5]}.json', 'w', encoding='utf-8') as out:
 	# 		json.dump(new_output, out, indent=4, ensure_ascii=False)
	#
	#
	#
	# for filename in os.listdir(path):
	# 	if (not filename.endswith('.json')) or (not filename.startswith('id')) or os.path.exists(f'clear_recheck/gpt3_remediation_output_English_Chinese_context_{filename[:-5]}.json'):
	# 		continue
	#
	# 	new_output = {'annotations': []}
	# 	print(filename)
	# 	with open(f'clear_recheck/{filename}') as out:
	# 		data = json.loads(out.read())
	#
	# 	for dialogue in tqdm(data['annotations']):
	# 		context = []
	# 		for index, c in enumerate(dialogue['dialogue'][:-1]):
	# 			if index % 2 == 0:
	# 				context.append(c['utterance'])
	# 			else:
	# 				context.append(c['translation'])
	# 		response = dialogue['dialogue'][-1]['utterance']
	# 		remediations = remediator.execute(context, response, dialogue['norm rule']['norm_type'])
	# 		corrections = corrector.execute(context, response, dialogue['norm rule']['norm_type'])
	#
	# 		final_correction = mini_BR.execute(corrections)
	# 		final_remediation = mini_BR.execute(remediations)
	# 		dialogue['gpt3_corrections'] = corrections
	# 		dialogue['gpt3_remediations'] = remediations
	# 		dialogue['gpt3_final_correction'] = final_correction
	# 		dialogue['gpt3_final_remediation'] = final_remediation
	#
	# 		new_output['annotations'].append(dialogue)
	#
	# 	with codecs.open(f'clear_recheck/gpt3_remediation_output_English_Chinese_context_{filename[:-5]}.json', 'w', encoding='utf-8') as out:
	# 		json.dump(new_output, out, indent=4, ensure_ascii=False)

