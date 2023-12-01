import rag

import time
import os
from rag import construct_context_for_junto
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY,)

def generate_debate(topic, context, left_house, right_house, conv_length):
	print("Generating Debate for {left_house}, {right_house}...")

	system_context = f"This is a debate between {left_house} and {right_house}. \
		The context for the topic of the debate is the following: {context}."

	arguments = 0
	debate = []

	while arguments < conv_length:
		argus = {}
		left_argument_context = f'{system_context}. Generate a 1 line argument \
			from speaker: {left_house} based on their past views and the provided context.'

		right_argument_context = f'{system_context}. Generate a 1 line argument \
		from speaker: {right_house} based on their past views and the provided context.'

		if len(debate) > 0:
			right_house_previous_argument = debate[arguments-1][right_house]
			left_argument_context += " Also, base your response on the past response \
				from the other speaker: {right_house_previous_argument}"

		left_house_argument = anthropic.completions.create(
			model="claude-2",
			max_tokens_to_sample=300,
			prompt=f"\n\nHuman: {left_argument_context}. \
				Do not use the name of the speaker: {left_house} in your response as part of the formatting. \
				\n\nAssistant:"
		)
		argus[left_house] = left_house_argument.completion

		right_argument_context += f" Additionally, try to respond to the other speaker's arugment: {left_house_argument}"

		right_house_argument = anthropic.completions.create(
			model="claude-2",
			max_tokens_to_sample=300,
			prompt=f"\n\nHuman: {right_argument_context}. \
				Do not use the name of the speaker: {left_house} in your response as part of the formatting. \
				\n\nAssistant:"
		)
		argus[right_house] = right_house_argument.completion

		debate.append(argus)

		arguments += 1
	return debate