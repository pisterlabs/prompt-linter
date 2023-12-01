'''Extracts source from the PLM generated analogies
Assign your open ai api key to openai.api_key
analogies_path should contain the path to the classified analogies from the analoginess scorer
src_path should contain the output path for saving the extracted source
''' 
import openai
import os 
from time import sleep
def extract_src(analogy,target):
	print(analogy)
	response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Table summarizing the following analogies:\nOne way to think of empirical risk minimization is as a process of tuning a machine learning model so that it performs well on the training data. The goal is to find a configuration of the model parameters that leads to the lowest possible error on the training set. This can be thought of as analogous to tuning a car’s engine so that it runs as smoothly as possible.\n| Target | Source\n| Empirical risk minimization | tuning a car’s engine\n'''\n"
  +analogy+
  "\n| Target | Source\n"
  +"| "+target+" |",
  temperature=0,
  max_tokens=50,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop = ["'''"]
)
	print(response)
	return response["choices"][0]["text"].strip('\n').replace('\n','###')



def read_f(path):
	lines = []
	targets = []
	prompts = []
	tmps = []
	domains = []
	pred_cls = []

	with open(path) as f:
		for row in f.readlines():
			splt = row.strip('\n').split('\t')
			lines.append(splt[0])
			targets.append(splt[1])
			prompts.append(splt[2])
			tmps.append(splt[3])
			domains.append(splt[4])
			pred_cls.append(splt[5])

	return lines,targets,prompts,tmps,domains,pred_cls


def main(analogies_path,src_path):
	analogies, targets, prompts, tmps, domains, pred_cls = read_f(analogies_path)
	srcs = []
	with open(src_path,'r') as f:
		danalogies, dtargets, dprompts, dtmps, ddomains, dpred_cls = read_f(src_path)
	done = list(zip(danalogies,dtmps,dprompts))
	with open(src_path,'a') as f:
		for idx,analogy in enumerate(analogies):
			print(idx)
			if (analogy,tmps[idx],prompts[idx]) not in done:
				if pred_cls[idx] == '1':
					sleep(0.8)
					src = extract_src(analogy,targets[idx])
					f.write(analogy+'\t'+targets[idx]+'\t'+prompts[idx]+'\t'+tmps[idx]+'\t'+domains[idx]+'\t'+pred_cls[idx]+'\t'+src+'\n')



if __name__ == '__main__':
	analogies_path = '../data/analoginess_scorer/non_adapt.txt'
	src_path = '../data/extracted_src/non_adapt.txt'
	openai.api_key = OPENAI_API_KEY
	main(analogies_path,src_path)
