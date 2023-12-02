from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
import os
import openai
from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
import logging

openai.api_key = os.environ["OPENAI_API_KEY"]
logging.basicConfig(level=logging.INFO)

prompt = """
Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.

Here are examples with input questions along with their expected outputs:

input=“Question: What is the capital city of France?"
output=“Paris”

input="Question: When was the American Declaration of Independence signed?"
output="July 4, 1776"

input="Question: What is the tallest mountain in the world?"
output="Mount Everest"
"""

prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
unlimited_dataset_generator = OpenAIDatasetGenerator(temperature=1.7, batch_size=5)
unlimited_dataset_generator.generate_dataset_split(
    prompt_spec, 200, split=DatasetSplit.TRAIN
)
