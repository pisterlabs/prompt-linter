'''Evaluates GP3 on RNPC tasks.'''

import os
import sys
os.chdir("../../../..")

root_dir = os.getcwd()
sys.path.append(f"{root_dir}/source")

# config
from configuration import Config
config_path = (f'source/Qa/eval_on_RNPC/gpt3/config.json')
config = Config.from_json_file(config_path)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
import torch

from Qa.utils import compute_scores, n_classes_dict_NP, label_text2id, label_id2text, task_fieldnames, read_dataset, query, unchanged_fields

import numpy as np
import csv
from utils import task_name_dict
import openai

from Qa.finetune_on_benchmark.gpt3.eval_gpt3_indomain import finetuned_model_dict


def map_class(task, pred_label, label_probs):
	'''Maps GPT3's predicted label to the RNPC task label.'''
	if task == "MPTE":
		assert pred_label in [0, 1, 2]
		entail_prob = label_probs[2]
		nonentail_prob = label_probs[0] + label_probs[1] # sum up contradiction & neutraal
		total_prob = entail_prob + nonentail_prob
		if entail_prob >= nonentail_prob:
			return 1, entail_prob/total_prob # entail
		else:
			return 0, nonentail_prob/total_prob # non-entail
	elif task == "EPC":
		assert pred_label in [0, 1, 2]
		total_prob = label_probs[0] + label_probs[1] + label_probs[2]
		pred_prob = label_probs[pred_label] / total_prob
		return pred_label, pred_prob

def convert_example_to_prompt(task, example):
	'''Converts an exmaple to GPT3 prompt.'''
	prompt = ""
	if task == "MPTE":
		premise, hypothesis = example["premise"], example["hypothesis"]
		prompt = f"Premises: {premise}\nHypothesis: {hypothesis}\n\n###\n\n"
	elif task == "EPC":
			event1, event2 = example["first_event"], example["second_event"]
			prompt = f"Event 1: {event1}\nEvent 2: {event2}\n\n###\n\n"
	if prompt == "":
		raise ValueError("Empty prompt.")
	return prompt

if __name__ == "__main__":

	# config
	task = eval(config.task)

	# not supporting SPTE, since the finetuned GPT3 models on SNLI/MNLI are not publicly available
	if task not in ["MPTE", "EPC"]:
		raise ValueError(f"Unspported task: {task}. Please choose from 'MPTE', and 'EPC'.")

	model = eval(config.gpt3_model)
	trial = config.trial


	# dirs
	frn = f"data/RNPC/tasks/{task}.csv"
	test_set = read_dataset(frn)

	pred_dir = f"output_dir/RNPC/eval_models_ft_on_benchmark/{task}"
	fwn = f"{pred_dir}/gpt3-{model}.csv"

	print(f"Evaluating models on {task}...")
	print(f"gpt3-{model}")

	gold_labels, pred_labels = [], []

	with open(fwn, 'a') as fw:
		writer = csv.DictWriter(fw, fieldnames=task_fieldnames[task])
		writer.writeheader()

		for test_id, test_example in enumerate(test_set["examples"]):

			# trial run on 5 examples
			if trial and test_id > 5:
				break

			if test_id % 50 == 0:
				print(f"{test_id} examples finished.")

			new_row = {}
			for field in unchanged_fields:
				if field in test_example:
					new_row[field] = test_example[field]

			# prediction
			prompt = convert_example_to_prompt(task, test_example)
			raw_pred_label, raw_label_probs = query(finetuned_model_dict[task][model], prompt)

			# map the predicted label to RNPC class
			pred_label_id, confidence = map_class(task, raw_pred_label, raw_label_probs)

			if pred_label_id == None:
				raise ValueError(f"Empty label. \nPrompt: {prompt}\n Raw prediction:{raw_pred_label}\n Raw probs:{raw_label_probs}")

			gold_label = test_example["label"]
			pred_label = label_id2text(pred_label_id, task)
			gold_labels.append(label_text2id(gold_label, task))
			pred_labels.append(pred_label_id)

			new_row["confidence"], new_row["gold label"], new_row["pred label"] = confidence, gold_label, pred_label
			writer.writerow(new_row)
			fw.flush()

	# compute accuracy, precision, recall, and f1
	scores = compute_scores(n_classes_dict_NP[task], gold_labels, pred_labels)
	print(scores)


