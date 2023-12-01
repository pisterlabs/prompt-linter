import ast
import json
from .nl_to_sql_prompts import (
    system_prompt_general, prompt_get_most_relevant_severity, system_prompt_sql,
    prompt_get_sql_query
)
from ..embeddings.chroma import chroma_openai_severity_collection
from ..llm.ai_services import gpt4
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def get_similar_severities(query, num_results=5):
	similar_severities = chroma_openai_severity_collection.query(
			query_texts=query, 
			n_results=num_results
		)
	return similar_severities

async def get_most_relevant_severity(inputs):
    system_message = SystemMessagePromptTemplate(
        prompt = system_prompt_general
    )
    human_message = HumanMessagePromptTemplate(
        prompt = prompt_get_most_relevant_severity
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message, 
            human_message
        ]
    )
    gpt_response = await gpt4.async_generate(
            chat_prompt, inputs
          )
    gpt_response = json.loads(gpt_response)
    most_relevant_severity = gpt_response["most relevant severity"]
    return most_relevant_severity

async def get_sql_query(inputs):
    system_message = SystemMessagePromptTemplate(
        prompt = system_prompt_sql
    )
    human_message = HumanMessagePromptTemplate(
        prompt = prompt_get_sql_query
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message, 
            human_message
        ]
    )
    gpt_response = await gpt4.async_generate(
        chat_prompt, inputs
    )
    gpt_response = get_json_from_text(gpt_response)
    sql_query = gpt_response["sql_query"]
    return sql_query

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
				json_to_dict = {"sql_query":""}
		return json_to_dict
	except Exception as e:
		print(f"Exception: {e}")
		return {"sql_query":""}