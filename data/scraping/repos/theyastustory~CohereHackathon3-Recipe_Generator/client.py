from annoy import AnnoyIndex
import cohere
from dotenv import load_dotenv
import numpy as np
import os
import re

from templates import prompt_header, prompt_item
from util import format_list, load_recipes, remove_one_recipe_from_prompt

# load api_key
load_dotenv()

# GLOBALS
API_KEY = os.getenv('API-KEY')
MODEL = 'large'
TRUNCATE = "LEFT"
RECIPES_FILE = './test_recipes.csv'
MAX_PROMPT_LEN = 2048
NUM_GEN_CHARS = 200
NUM_NEIGHBOURS = None # default to entire dataset

# init client
co = cohere.Client(API_KEY)
recipes = load_recipes(RECIPES_FILE)
ingredients = [format_list(ings) for ings in recipes.ingredients]

# compute embeddings
embeddings = np.array(co.embed(model=MODEL,texts=ingredients, truncate=TRUNCATE).embeddings)


"""
Search index for nearest neighbor semantic search
"""
# Create the search index, pass the size of embedding
search_index = AnnoyIndex(embeddings.shape[1], 'angular')
# Add all the vectors to the search index
for i in range(embeddings.shape[0]):
	search_index.add_item(i, embeddings[i])

search_index.build(10) # 10 trees


"""
Query Embedding (from user input)
"""

def get_nns_from_query(query):
	"""
	take query as input, embed, and return similar indices from recipes
	"""
	query_embedding = co.embed(texts=[query], model=MODEL, truncate=TRUNCATE).embeddings[0]
  
	similars_ids, _ = search_index.get_nns_by_vector(
		query_embedding, 
		n=NUM_NEIGHBOURS if NUM_NEIGHBOURS else len(embeddings), 
		include_distances=True
  	)
  
	return similars_ids



"""
Generating
"""
def build_prompt_from_similars(similar_ids, query, n=10):
  
	prompt = prompt_header
	similar_recipes = recipes.iloc[similar_ids[:n]]
  
	for _, (ings, steps, name) in similar_recipes.iterrows():
		prompt += prompt_item.format(format_list(ings), format_list(steps), re.sub(' +', ' ', name))
	
	prompt += f"Ingredients:{query}"
   
	return prompt


def generate_recipe(prompt):
	"""
	Generate recipe from cohere API. If query is too long,
	delete last recipe
	"""
	while True:
		try:
			response = co.generate(
				model=MODEL,
				prompt=prompt,
				max_tokens=200,
				temperature=1,
				k=3,
				p=0.75,
				frequency_penalty=0,
				presence_penalty=0,
				stop_sequences=['--'],
				return_likelihoods='NONE'
			)
			return response.generations
		
		except cohere.error.CohereError:
			prompt = remove_one_recipe_from_prompt(prompt)


def generate_from_query(query):
	"""
	Function to implement logic of this module end-to-end
	"""
	similar_ids = get_nns_from_query(query)
	prompt = build_prompt_from_similars(similar_ids, query)
	generations = generate_recipe(prompt)

	return generations