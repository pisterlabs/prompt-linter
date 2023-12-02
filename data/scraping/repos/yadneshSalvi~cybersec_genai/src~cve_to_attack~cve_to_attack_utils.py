import re
import ast
import json
import tiktoken
from langchain.prompts import PromptTemplate
from ..embeddings.chroma import chroma_openai_cwe_collection, chroma_openai_attack_collection

tokenizer = tiktoken.get_encoding("cl100k_base")

general_cyber_security_prompt = PromptTemplate(
	input_variables=["query"],
	template="""
You are a cyber-security expert and will answer the following question.
Question: '''{query}'''
"""
	)

cve_to_attack_prompt = PromptTemplate(
	input_variables=["prompt"],
	template="""
{prompt}
"""
	)

def get_required_cve_data(similar_cves):
	metadatas = similar_cves['metadatas'][0]
	documents = similar_cves['documents'][0]
	top_cves_match = [] 
	descriptions = [] 
	techniques = [] 
	token_lens_cve = []
	for metadata, description in zip(metadatas, documents):
		top_cves_match.append(metadata['cve'])
		descriptions.append(description)
		techniques_list = [i.replace('\'','').replace(
				'\"','').strip() for i in metadata['technique'].split(',')]
		techniques.append(techniques_list)
		token_lens_cve.append(metadata['cve_description_token_len'])
	return top_cves_match, descriptions, techniques, token_lens_cve

def get_required_technique_data(similar_techniques):
	metadatas = similar_techniques['metadatas'][0]
	documents = similar_techniques['documents'][0]
	top_techniques_match = []
	descriptions = []
	token_lens_technique = []
	for metadata, description in zip(metadatas, documents):
		techniques_list = [i.replace('\'','').replace(
				'\"','').strip() for i in metadata['technique'].split(',')]
		top_techniques_match.extend(techniques_list)
		descriptions.append(description)
		token_lens_technique.append(metadata['technique_description_token_len'])
	return top_techniques_match, descriptions, token_lens_technique

def get_cve_examples(top_cves_match, descriptions, techniques):
	cve_similarity_prediction = []
	for technique in techniques:
		for t in technique:
			if t not in cve_similarity_prediction:
				cve_similarity_prediction.append(t)
	example_cves = ""
	for i, j, k in zip(top_cves_match, descriptions, techniques):
		j = j.replace('\n',' ')
		json_format_ = "{\"related_attacks\": " + str(k) + "}"
		example_cves += f"{i}: {j}\n{json_format_}\n"
	return example_cves, cve_similarity_prediction

def get_similar_cves(query, num_results=5):
	similar_cves = chroma_openai_cwe_collection.query(
			query_texts=query, 
			n_results=num_results
		)
	return similar_cves

def get_similar_techniques(query):
	similar_techniques = chroma_openai_attack_collection.query(
			query_texts=query, 
			n_results=10
		)
	return similar_techniques

def remove_links_citations(text):
	# use .*? for non-greedy regex
	regex_links = r"\(https?.*?\)" # all links were present inside round brackets
	text = re.sub(regex_links, '', text, flags = re.I)
	regex_cite = r"\(Citation:.*?\)" # all citations were also present inside round brackets
	text = re.sub(regex_cite,'',text,flags=re.I)
	text = " ".join(text.split())
	return text

def get_len_token(text):
	tokens = tokenizer.encode(str(text))
	num_ip_tokens = len(tokens)
	return num_ip_tokens

def make_few_shot_prompt(cve_description, attack_descriptions, example_cves, json_format):
	prompt = f"""
You have been given an user search string below and the possible attack descriptions that it can be related to.
Your task is to find out the exact attack descriptions that the user search string can map to.
You can make use of the examples given below.

User search string: '''{cve_description}'''

Attack descriptions that above user search string can map to:
'''
{attack_descriptions}
'''

Examples of similar CVEs and the attack descriptions that they exactly map to:
'''
{example_cves}
'''

You should first write down the CVEs from the examples that are most similar to the user search string.
Then you should write down the reasons why the most similar CVEs from examples are mapped to the attack descriptions provided in the examples.
Now based on this information, attack descriptions and using critical reasoning map the given user search string with the exact attack descriptions, and write down the reasoning for it.
Finally fill the below json and wite it down in correct json format:
{json_format}
"""
	return prompt

def create_prompt_in_token_limit(
		cve_description, attacks_1, attacks_2, example_cves, json_format
	):
	attacks_combined = []
	for att in attacks_1:
		if att not in attacks_combined:
			attacks_combined.append(att)
	for attack in attacks_2:
		if attack not in attacks_combined:
			attacks_combined.append(attack)
	
	prompt_initial = make_few_shot_prompt(cve_description, '', example_cves, json_format)
	prompt_len = get_len_token(prompt_initial)

	i=1
	while prompt_len < 4000 and i<len(attacks_combined):
		attack_descriptions = ""
		for technique in attacks_combined[:i]:
			technique_description = chroma_openai_attack_collection.get(
										where={"technique": technique},
									)
			technique_description = technique_description['documents'][0]
			technique_description = remove_links_citations(technique_description)
			attack_descriptions += f"{technique}: {technique_description}\n"
		prompt = make_few_shot_prompt(
			cve_description, attack_descriptions, example_cves, json_format
		)
		prompt_len = get_len_token(prompt)
		i+=1
	return prompt

def get_json_from_text(text):
	try:
		text_reversed = text[::-1]
		for idx, t in enumerate(text_reversed):
			if t == '}':
				last_bracket = len(text) - idx
			elif t == '{':
				first_bracket = len(text) -1 -idx
				break
		json_text = text[first_bracket:last_bracket]
		try:
			json_to_dict = json.loads(json_text)
		except:
			try:
				json_to_dict = ast.literal_eval(json_text)
			except:
				json_to_dict = {"related_attacks":[]}
		return json_to_dict
	except Exception as e:
		print(f"Exception: {e}")
		return {"related_attacks":[]}

def make_cve_to_attack_prompt(query):
	similar_cves = get_similar_cves(query)
	top_cves_match, cve_descriptions, \
	techniques, token_lens_cve = get_required_cve_data(similar_cves)
	cve_descriptions = [remove_links_citations(i) for i in cve_descriptions]
	example_cves, cve_similarity_prediction = get_cve_examples(
		top_cves_match, cve_descriptions, techniques)
	
	similar_techniques = get_similar_techniques(query)
	top_techniques_match, technique_descriptions, token_lens_technique = get_required_technique_data(
		similar_techniques)
	technique_descriptions = [remove_links_citations(i) for i in technique_descriptions]

	json_format = "{\"related_attacks\":[]}"
	cve_description = query #f"{question_cve}: {question_cve_description}"

	prompt = create_prompt_in_token_limit(
			cve_description, cve_similarity_prediction, 
			top_techniques_match, example_cves, json_format
		)
	return cve_to_attack_prompt, prompt

def search_similar_cves(query, num_results):
	similar_cves = get_similar_cves(query, num_results)
	top_cves_match, cve_descriptions, \
	techniques, token_lens_cve = get_required_cve_data(similar_cves)
	response = [
		{
			"cve_name":top_cves_match[i],
			"cve_description":cve_descriptions[i],
			"attack_techniques": techniques[i]
		} for i in range(len(top_cves_match))
	]
	return response
	
def search_similar_cves_with_technique_descp(query, num_results):
	similar_cves = get_similar_cves(query, num_results)
	top_cves_match, cve_descriptions, \
	techniques, token_lens_cve = get_required_cve_data(similar_cves)
	response = [
		{
			"cve_name":top_cves_match[i],
			"cve_description":cve_descriptions[i],
			"attack_techniques":[
				{
					"attack_technique_name":technique,
					"attack_technique_description":chroma_openai_attack_collection.get(
											where={"technique": technique}
										)["documents"][0]
				} for technique in techniques[i]
			] 
		} for i in range(len(top_cves_match))
	]
	return response