import streamlit as st
import wikipedia
import openai
import pymongo
import tempfile
import certifi
import gridfs
from langchain.agents import tool
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.vectorstores import Chroma,FAISS
from typing import List, Dict
import requests
import json
import configparser
import os
config = configparser.ConfigParser()
config.read('config.ini')
db_host = st.secrets["db_host"]
#db_host = config['constants']['db_host']
# serper_key = st.secrets["serpapi"]
# bing_key = st.secrets["bingapi"]
# serper_url = config['constants']['serp_url']
# bing_url = config['constants']['bing_url']
db_client = config['constants']['db_client']
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())
db = client[db_client]
data_collection = db[config['constants']['sd']]
user_info_collection = db[config['constants']['ui']]



def get_wikipedia_data(query, max_results=4):
	try:
		search_results = wikipedia.search(query)[:max_results]
		search_results_with_summary_and_url = []
		st.session_state.tool_use = True
		st.session_state.source_bot == False
		st.session_state.related_questions = []

		for i, result in enumerate(search_results):
			try:
				page = wikipedia.page(result)
				summary = wikipedia.summary(result, sentences=10)
				if i == 0:
					st.session_state.web_link.append(f'Ref No: {st.session_state.s_count} - {result}')
					st.session_state.web_link.append(page.url)
					search_results_with_summary_and_url.append({
						"title": result,
						"url": page.url,
						"summary": summary
					})
				else:
					summary = wikipedia.summary(result, sentences=1)
					question_info = {
						"question": result,
						"snippet": summary,
						"url": page.url
					}
					st.session_state.related_questions.append(question_info)
			except Exception as e:
				st.warning(f"Unable to find Wikipedia resources for {result}: {str(e)}")

		data = {
			"query": query,
			"search_results": search_results_with_summary_and_url
		}
		return data
	except Exception as e:
		st.write(f"Error: {str(e)}")
		return None

@tool
def wikipedia_to_json_string(query: str) -> str:
	"""
	Searches Wikipedia for the given query and returns the search results
	with their URLs as a JSON formatted string.
	"""
	data = get_wikipedia_data(query)
	if data:
		json_string = json.dumps(data, ensure_ascii=False, indent=4)
		return json_string
	else:
		return None


# @tool
# def google_search_crawl(query: str) -> str:
# 	"""
# 	Searches Google for the given query and returns the search results
# 	with their titles, URLs, and descriptions as a JSON formatted string.
# 	"""
# 	# Prepare the actor input
# 	client = ApifyClient(apify_key)

# 	st.session_state.tool_use = True
# 	st.session_state.source_bot == False
# 	st.session_state.web_link.append(f'Ref No: {st.session_state.s_count}')
# 	run_input = {
# 		"queries": query,
# 		"maxPagesPerQuery": 1,
# 		"resultsPerPage": 2,
# 		"countryCode": "sg",
# 		"customDataFunction": """async ({ input, $, request, response, html }) => {
# 		  return {
# 			pageTitle: $('title').text(),
# 		  };
# 		};""",
# 	}

# 	# Run the actor and wait for it to finish
# 	run = client.actor("apify/google-search-scraper").call(run_input=run_input)

# 	# Fetch the dataset items
# 	dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items

# 	# Extract organic results from the dataset items
# 	organic_results = []
# 	for item in dataset_items:
# 		if "organicResults" in item:
# 			organic_results = item["organicResults"]
# 			break

# 	formatted_results = []
# 	for result in organic_results:
# 		formatted_result = {"title": result["title"], "url": result["url"], "summary": result["description"]}
# 		formatted_results.append(formatted_result)

# 		# Append the result title and url to st.session_state.web_link
# 		st.session_state.web_link.append(f'Ref No: {st.session_state.s_count} - {result["title"]}')
# 		#st.session_state.web_link.append(result["title"])
# 		st.session_state.web_link.append(result["url"])

# 	# Create a dictionary with the query and search_results
# 	data = {
# 		"query": query,
# 		"search_results": formatted_results
# 	}

# 	# Convert the data dictionary to a JSON string
# 	json_string = json.dumps(data, ensure_ascii=False, indent=4)

# 	return json_string

def extract_organic_results_and_people_also_ask(response_data: dict) -> dict:
	organic_results = response_data.get('organic', [])
	first_result = organic_results[0] if len(organic_results) > 0 else {}
	second_result = organic_results[1] if len(organic_results) > 1 else {}

	title = first_result.get('title', '')
	link = first_result.get('link', '')

	st.session_state.web_link.append(f'Ref No: {st.session_state.s_count} - {title}')
	st.session_state.web_link.append(link)

	snippet_1 = first_result.get('snippet', '')
	snippet_2 = second_result.get('snippet', '')

	combined_snippet = f"{snippet_1} {snippet_2}"

	people_also_ask = response_data.get('peopleAlsoAsk', [])
	st.session_state.related_questions = []
	for question_data in people_also_ask:
		question = question_data.get('question', '')
		snippet = question_data.get('snippet', '')
		link = question_data.get('link', '')
		question_info = {
			"question": question,
			"snippet": snippet,
			"url": link
		}
		st.session_state.related_questions.append(question_info)

	return {
		"title": title,
		"url": link,
		"combined_snippet": combined_snippet
	}

@tool
def google_search_serp(query: str) -> str:
	"""
	Searches Google for the given query and returns the search results
	with their titles, URLs, and descriptions as a JSON formatted string.
	"""
	st.session_state.tool_use = True
	st.session_state.source_bot == False
	payload = json.dumps({
	  "q": query,
	  "gl": "sg"
	})
	headers = {
	  'X-API-KEY': serper_key,
	  'Content-Type': 'application/json'
	}

	response = requests.request("POST", serper_url, headers=headers, data=payload)
	response_data = json.loads(response.text)
	#st.write(response_data)
	#st.write(type(response_data))
	final_data = extract_organic_results_and_people_also_ask(response_data)
	#st.write(st.session_state.related_questions)
	#st.write(knowledge_graph_data)
	# Create a dictionary with the query and search_results
	data = {
		"query": query,
		"search_results": [final_data]
	}

	json_string = json.dumps(data, ensure_ascii=False, indent=4)

	return json_string


def extract_bing_results(response_data: dict) -> dict:
	organic_results = response_data.get('webPages', {}).get('value', [])
	first_result = organic_results[0] if len(organic_results) > 0 else {}
	second_result = organic_results[1] if len(organic_results) > 1 else {}

	title = first_result.get('name', '')
	displaylink = first_result.get('displayUrl', '')
	#displayUrl = first_result.get('displayUrl', '')

	st.session_state.web_link.append(f'Ref No: {st.session_state.s_count} - {title}')
	st.session_state.web_link.append(displaylink)

	snippet_1 = first_result.get('snippet', '')
	snippet_2 = second_result.get('snippet', '')

	combined_snippet = f"{snippet_1} {snippet_2}"

	related_searches = response_data.get('relatedSearches', {}).get('value', [])
	
	st.session_state.related_questions = []
	for i, question_data in enumerate(related_searches[:3]):
		question = question_data.get('text', '')
		displayText = question_data.get('displayText', '')
		link = question_data.get('webSearchUrl', '')
		question_info = {
			"question": question,
			"snippet": displayText,
			"url": link
		}
		st.session_state.related_questions.append(question_info)

	return {
		"title": title,
		"url": displaylink,
		"combined_snippet": combined_snippet
	}



@tool
def bing_search_internet(query: str) -> str:
	"""
	Searches Bing internet search for the given query and returns the search results
	with their titles, URLs, and descriptions as a JSON formatted string.
	"""


	st.session_state.tool_use = True
	st.session_state.source_bot == False
	# Add your Bing Search V7 subscription key and endpoint to your environment variables.
	#subscription_key = os.environ[bing_key]
	subscription_key = bing_key
	assert subscription_key
	# Query term(s) to search for. 
	# Construct a request
	mkt = 'en-SG'
	fil = 'strict'

	params = { 'q': query, 'mkt': mkt, 'safeSearch': fil}
	headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

	# Call the API
	try:
		response = requests.get(bing_url, headers=headers, params=params)
		response.raise_for_status()
		response_data = response.json()
	except Exception as ex:
		raise ex
	
	#response_data = json.loads(response.text)
	#st.write(response_data)
	#st.write(type(response_data))
	final_data = extract_bing_results(response_data)
	#st.write(st.session_state.related_questions)
	#st.write(knowledge_graph_data)
	# Create a dictionary with the query and search_results
	data = {
		"query": query,
		"search_results": [final_data]
	}

	json_string = json.dumps(data, ensure_ascii=False, indent=4)

	return json_string

# @st.cache_resource
# def extract_files_from_mongodb(_tch_code):
# 	# Connect to MongoDB
# 	fs = gridfs.GridFS(db)
# 	# Create a temporary directory called tch_code
# 	temp_dir = tempfile.mkdtemp(prefix=_tch_code)

# 	# Get all the files associated with the given tch_code
# 	files = fs.find({"tch_code": _tch_code})

# 	# Write the files to the temporary directory
# 	for file in files:
# 		# Recreate the directory structure using the relative path metadata
# 		file_path = os.path.join(temp_dir, file.relative_path)
# 		os.makedirs(os.path.dirname(file_path), exist_ok=True)
		
# 		with open(file_path, "wb") as f:
# 			f.write(file.read())

# 	return temp_dir

# @st.cache_resource
# def extract_files_from_mongodb(_tch_code):
#     # Connect to MongoDB
#     fs = gridfs.GridFS(db)

#     # Create a temporary directory called tch_code
#     temp_dir = tempfile.mkdtemp(prefix=_tch_code)

#     # Get all the files associated with the given tch_code
#     files = fs.find({"tch_code": _tch_code})

#     # Write the files to the temporary directory
#     for file in files:
#         file_path = os.path.join(temp_dir, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(file.read())

#     return temp_dir

@st.cache_resource
def load_instance_index(_tch_code):
	embeddings = OpenAIEmbeddings()
	#vectordb = Chroma(collection_name=st.session_state.teacher_key, embedding_function=embeddings, persist_directory=_temp_dir)
	#vectordb = Pinecone.from_existing_index(st.secrets["pine_index"], embeddings, _tch_code)
	vectordb = FAISS.load_local(_tch_code, embeddings)

	return vectordb

@tool
def document_search(query: str) ->str:
	"""
	Searches vectorstore for anything related to class notes and materials for the given query in the topic and returns the search results
	with their titles, URLs, and descriptions as a JSON formatted string.
	"""
	openai.api_key  = st.session_state.api_key
	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
	os.environ["LANGCHAIN_HANDLER"] = "langchain" 
	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key

	docsearch = load_instance_index(st.session_state.teacher_key)
	retriever = docsearch.as_retriever(search_type="mmr")
	source_documents = retriever.get_relevant_documents(query)
	if source_documents:
		first_doc = source_documents[0]
		f_source = first_doc.metadata['source']
		f_topic = first_doc.metadata['topic']
		f_content = first_doc.page_content
		f_url = first_doc.metadata['url']
		
		st.session_state.web_link.append(f"Ref No: ({st.session_state.s_count})-{f_source}: {f_topic}")
		st.session_state.web_link.append(f_url)

	st.session_state.related_questions = []
	if source_documents:
		for document in source_documents:
			source = document.metadata['source']
			topic = document.metadata['topic']
			page = document.metadata['page']
			page_content = document.page_content
			
			st.session_state.related_questions.append(f"Ref No: ({st.session_state.s_count})-{source}: {topic}, Content page {int(page) + 1}")
			st.session_state.related_questions.append(page_content)

		st.session_state.tool_use = True
		st.session_state.source_bot = True


	# Format the result
	formatted_result = {
		"title": f_topic,
		"url": f_url,
		"summary": f_content
	}

	# Create a dictionary with the query and search_results
	data = {
		"query": query,
		"search_results": [formatted_result]
	}

	# Convert the data dictionary to a JSON string
	json_string = json.dumps(data, ensure_ascii=False, indent=4)

	return json_string

