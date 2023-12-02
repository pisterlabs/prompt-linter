import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import openai
import tiktoken

from settings import Settings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "test-pseudocode"
CODE_BASE_PATH = '../code-base/{engine}/{ques_id}.text'
CODE_UPDATED_PATH = '../code-updated/{engine}/{ques_id}.text'
SLEEP_TIME = 5
MAX_TOKENS = 4097
TEMPERATURE = 0.3

openai.api_key = OPENAI_API_KEY


def generate_para_steps_prompt(data) -> str:
    para_steps = ", ".join([p.strip() for p in data["question"]["para_steps"] if len(p) > 0])
    para_steps_prompt = f"The following procedure is described:\n{para_steps}" \
                        "Can you provide a Pseudocode representation of this procedure?"
    return para_steps_prompt


def generate_stem_prompt(data) -> str:
    stem = data['question']['stem'].strip()
    stem_prompt = f'Update the proposed Pseudocode based on : "{stem}"'
    return stem_prompt


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


with open(Settings.wiqa_test_path, 'r') as json_file:
    test_data = [json.loads(line) for line in json_file]

for data in test_data:
    ques_id = data['metadata']['ques_id']
    code_base_representation_path = Path(CODE_BASE_PATH.format(engine=MODEL_NAME, ques_id=ques_id))
    code_updated_representation_path = Path(CODE_UPDATED_PATH.format(engine=MODEL_NAME, ques_id=ques_id))

    if not code_base_representation_path.is_file():
        para_steps_prompt = generate_para_steps_prompt(data)
        stem_prompt = generate_stem_prompt(data)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": para_steps_prompt},
            ],
            temperature=0,
        )
        code_representation = response['choices'][0]['message']['content']

        with open(code_base_representation_path, 'w') as f:
            f.write(code_representation)
        print(ques_id, "code_representation finished")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": para_steps_prompt},
                {"role": "assistant", "content": code_representation},
                {"role": "user", "content": stem_prompt},
            ],
            temperature=0,
        )

        code_updated_representation = response['choices'][0]['message']['content']

        with open(code_updated_representation_path, 'w') as f:
            f.write(code_updated_representation)
        print(ques_id, "code_updated_representation finished")

        # Avoid hitting rate limit
        time.sleep(SLEEP_TIME)
