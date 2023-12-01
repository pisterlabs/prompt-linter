import random
import os
import openai
import argparse
from time import sleep
import string 

prompts = ['Explain <target> using an analogy.','Create an analogy to explain <target>.','Using an analogy, explain <target>.','What analogy is used to explain <target>?','Use an analogy to explain <target>.']

def perm(target):

	tlst = list(range(1,len(target)-1))
	random.shuffle(tlst)
	perm_pos1 = tlst[0]
	# perm_pos2 = tlst[1]
	perm_pos2 = perm_pos1+1
	if perm_pos2 == len(target)-1:
		perm_pos2 = perm_pos1 - 1
	target_copy = list(target)
	tmp = target_copy[perm_pos1]
	target_copy[perm_pos1] = target_copy[perm_pos2]
	target_copy[perm_pos2] = tmp 
	target = ''.join(target_copy)
	# print(perm_pos1,perm_pos2,target)
	return target


def del_(target):
	tlst = list(range(1,len(target)-1))
	random.shuffle(tlst)
	del_pos = tlst[0]
	# print(del_char,target[:del_char]+target[del_char+1:])
	return target[:del_pos]+target[del_pos+1:]

def ins(target):
	tlst = list(range(1,len(target)-1))
	random.shuffle(tlst)
	ins_pos = tlst[0]
	ins_char = random.choice(string.ascii_lowercase)
	# print(ins_char,target[:ins_pos]+ins_char+target[ins_pos:])
	return target[:ins_pos]+ins_char+target[ins_pos:]

def rep(target):
	tlst = list(range(1,len(target)-1))
	random.shuffle(tlst)
	rep_pos = tlst[0]
	alp = string.ascii_lowercase.replace(target[rep_pos],'')
	rep_char = random.choice(string.ascii_lowercase)
	target_copy = list(target)
	target_copy[rep_pos] = rep_char 
	target = ''.join(target_copy)
	print(rep_char,target,rep_pos)
	return target

def gen_anlgy(prompt,j,op,target,ntype):
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
		with open(op+'p'+str(j)+'_lt_'+ntype+'.txt','a') as f:
				for r in write_out:
					f.write(r)
				write_out = []

def main(op):

	ntype = 'rep' 
	concept_path = '../data/target_concepts/science.txt'
	

	with open(concept_path,'r') as f:
		data = f.readlines()

	write_out = []
	anlgies = {}
	targets = list()
	lencnt = 0

	for i,row in enumerate(data):
		row = row.strip('\n')
		if row.split('\t')[1] not in targets:
			targets.append(row.split('\t')[1])

	
	for i,target in enumerate(list(targets)):
		
		print(target)
		orig_target = target 
		if len(target)>3:
			target = rep(target)
		else:
			lencnt +=1

		done_bef = False
		try:
			anlgies[orig_target.lower()]
			done_bef = True 
		except KeyError as e:
			anlgies[orig_target.lower()] = []
		
		print(i)
		if not (done_bef) and i>=0:
			for j,prompt in enumerate(prompts):
				prompt = prompt.replace('<target>',target.lower())
				gen_anlgy(prompt,j,op,orig_target,ntype)

		anlgies[orig_target.lower()].append(orig_target)

	print(lencnt)
if __name__ == '__main__':
	openai.api_key = OPEN_API_KEY	
	outpath_prefix = '../data/noise/'

	main(outpath_prefix)

	# rep('nadh')


