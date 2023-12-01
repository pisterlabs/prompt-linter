#Generates various types of candidate analogies
#Replace OPENAI_API_KEY with your OpenAi API key

import os
import openai
import argparse
from time import sleep

prompts_wsrc = ['Explain <target> using an analogy involving <src>.','Explain how <target> is analogous to <src>.','Explain how <target> is like <src>.','Explain how <target> is similar to <src>.','How is <target> analogous to <src>?','How is <target> like <src>?','How is <target> similar to <src>?']


prompts_nanlgy = ['Explain <target>.','What is <target>?','Explain <target> in plain language to a second grader.']

prompts_nadpt_sci = ['Explain <target> using an analogy.','Create an analogy to explain <target>.','Using an analogy, explain <target>.','What analogy is used to explain <target>?','Use an analogy to explain <target>.']

prompts_nadpt_ai = ['Explain <target> (machine learning) using an analogy.','Create an analogy to explain <target> (machine learning).','Using an analogy, explain <target> (machine learning).','What analogy is used to explain <target> (machine learning)?','Use an analogy to explain <target> (machine learning).']

prompts_nadpt_cyber = ['Explain <target> (cybersecurity) using an analogy.','Create an analogy to explain <target> (cybersecurity).','Using an analogy, explain <target> (cybersecurity).','What analogy is used to explain <target> (cybersecurity)?','Use an analogy to explain <target> (cybersecurity).']

discps = ['a business','an arts','a social science']
prefs = ['music','sports','cooking','gardening']

prompts_discp = ['Using an analogy, explain <target> to <a discp> student.','Explain <target> using <a discp> analogy.', 'Explain <target> using an analogy. The analogy should be suitable for <a discp> student.', 'Using <a discp> analogy, explain <target>.','What is a good analogy to explain <target> to <a discp> student?']

prompts_pref = ['Explain <target> using a <pref> analogy.', 'Explain <target> using an analogy. The analogy should be suitable for student who prefs <pref>.', 'Using a <pref> analogy, explain <target>.','What is a good analogy to explain <target> to student who prefs <pref>?','Using an analogy, explain <target> to student who likes <pref>.']

def gen_anlgy(prompt,j,op,target):
	write_out = []
	if j >=0:
		for k in range(1):
			if k>=0:
				sleep(0.8)
				print(prompt)
				pred = openai.Completion.create(
				engine="text-davinci-001",
				prompt= prompt,
				temperature=0.85,
				max_tokens=939,
				top_p=1, 
				best_of=1,
				frequency_penalty=1.24,
				presence_penalty=1.71
				)
				print(j,k,prompt)
				print(pred)

			write_out.append(pred["choices"][0]["text"].replace('\n','')+'\t'+target+'\t'+prompt+'\n')
			with open(op+'_p'+str(j)+'_ht.txt','a') as f:
				for r in write_out:
					f.write(r)
				write_out = []
	if j>=0:
		pred = openai.Completion.create(
		engine="text-davinci-001",
		  prompt= prompt,
		  temperature=0,
		  max_tokens=939,
		  top_p=1,
		  best_of=1,
		  frequency_penalty=0,
		  presence_penalty=0
		)
		print(j,prompt)
		print(pred)

		write_out.append(pred["choices"][0]["text"].replace('\n','')+'\t'+target+'\t'+prompt+'\n')
		with open(op+'_p'+str(j)+'_ht.txt','a') as f:
				for r in write_out:
					f.write(r)
				write_out = []

def main(pt,domain,op):

	if pt == '1':
		if domain == '1':
			prompts = prompts_nadpt_sci
			concept_path = '../data/target_concepts/science.txt'
		elif domain == '2':
			prompts = prompts_nadpt_cyber
			concept_path = '../data/target_concepts/cyber.txt'
		else:
			prompts = prompts_nadpt_ai 
			concept_path = '../data/target_concepts/ai.txt'
	else:
		concept_path = '../data/target_concepts/science.txt'
		domain = '1'
		
		if pt == '2':
			prompts = prompts_pref

		elif pt == '3':
			prompts = prompts_discp
		elif pt == '4':
			prompts = prompts_nanlgy
		else:
			prompts = prompts_wsrc 


	with open(concept_path,'r') as f:
		data = f.readlines()

	write_out = []
	anlgies = {}
	for i,row in enumerate(data):
		row = row.strip('\n')
		if domain != '1':
			target = row.strip()
		else:
			row = row.split('\t')
			target = row[1].strip()
			src = row[0].strip()
		
		done_bef = False
		try:
			anlgies[target.lower()]
			done_bef = True 
		except KeyError as e:
			anlgies[target.lower()] = []
		
		print(i)
		if (pt == '5' or not done_bef) and i>=0:
			for j,prompt in enumerate(prompts):
				prompt = prompt.replace('<target>',target.lower())
				if pt == '2':
					for l,pref in enumerate(prefs):
						if l>=0:
							mprompt = prompt
							mprompt = mprompt.replace('<pref>',pref)
							gen_anlgy(mprompt,j,op,target)
				elif pt == '3':
					for d,discp in enumerate(discps):
						if d>=0:
							mprompt = prompt
							mprompt = mprompt.replace('<a discp>',discp)
							gen_anlgy(mprompt,j,op,target)
				elif pt == '5':
					prompt = prompt.replace('<src>',src.lower())
					gen_anlgy(prompt,j,op,target)
				else:
					gen_anlgy(prompt,j,op,target)

		anlgies[target.lower()].append(target)


if __name__ == '__main__':
	openai.api_key = OPENAI_API_KEY 
	parser = argparse.ArgumentParser(description='Enter configuration')
	parser.add_argument('--prompt_type', metavar='pt', type=str,
		help='Type of prompts. Enter 1 for non-adaptive, 2 for preference-specific adaptive, 3 for discipline-specific adaptive, 4 for non-analogies, 5 for analogies with given sources.',required=True)
	parser.add_argument('--domain', metavar='d', type=str,
		help='Domain for non-adaptive analogies. Enter 1 for Science, 2 for Cybersecurity, 3 for Machine Learning.',required=False)
	parser.add_argument('--outpath_prefix', metavar='op', type=str,
		help='Prefix of the output paths.',required=True)
	

	args = parser.parse_args()
	if args.prompt_type == '1' and (args.domain is None or args.domain not in ['1','2','3']):
		print('Please enter valid domain')
		exit()
	elif args.prompt_type not in ['1','2','3','4','5']:
		print('Please enter valid prompt type')
		exit()

	main(args.prompt_type,args.domain,args.outpath_prefix)


	