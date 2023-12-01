import json
import os

import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
	ChatPromptTemplate,
	HumanMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotPromptTemplate

os.environ["OPENAI_API_KEY"] = ""
openai.organization = ""


def generate_summaries(chat, dialogues):
	with open("test_pos_7_prompts.json") as f:
		few_shot_pos_7 = json.load(f)
	generated_section_text_list = []
	for idx, dialogue in enumerate(dialogues):
		print("Processing {} sample".format(idx + 1))
		try:
			example_prompt = PromptTemplate(input_variables=["dialogue", "summary"],
											template="Dialogue:\n{dialogue}\n\nSummary:\n{summary}")
			prompt = FewShotPromptTemplate(examples=few_shot_pos_7[idx], example_prompt=example_prompt,
										   suffix="Dialogue: {input}\n\nSummary:\n", input_variables=["input"])
			human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)
			chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
			generations = chat.generate([chat_prompt.format_prompt(input=dialogue).messages]).generations
			generated_section_text = generations[0][0].text
			generated_section_text_list.append(generated_section_text)
		except Exception as E:
			print("#### Exception ####")
			print(E)
			print("####")
			continue
	return generated_section_text_list


def run_task_A_summarization(dialogues):
	chat = ChatOpenAI(model_name='gpt-4', temperature=0., max_tokens=800)
	generated_section_text_list = generate_summaries(chat, dialogues)
	generated_section_text_list = [x.strip().replace('\n', ' ').replace('\r', '') for x in generated_section_text_list]
	return generated_section_text_list
