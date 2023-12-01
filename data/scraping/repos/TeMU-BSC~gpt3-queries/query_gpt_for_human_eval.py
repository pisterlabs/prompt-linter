import os
import openai
import json
from dataclasses import dataclass
import time
import git
import uuid
from tqdm import tqdm


ENGINE = 'all'
TEMPERATURE = 0.7
MAX_TOKENS = 512
TOP_P = 0.95
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
FILES = ['ca.json', 'en.json']
ENGINES = ['ada', 'babbage', 'curie', 'davinci']

UPPER_MAX_TOKENS = 2048

@dataclass
class GPTConfig:
    engine: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float


config = GPTConfig(engine=ENGINE,
                   temperature=TEMPERATURE,
                   max_tokens=MAX_TOKENS,
                   top_p=TOP_P,
                   frequency_penalty=FREQUENCY_PENALTY,
                   presence_penalty=PRESENCE_PENALTY)

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_gpt_answer(prompt, engine) -> str:
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    answer = response.last_response.data['choices'][0]['text']

    return answer


if __name__ == '__main__':
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    output_path = 'output'
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    extra_id = uuid.uuid4().hex
    output_directory = os.path.join(output_path, f'sample-{config.engine}-{timestamp}-{sha[:4]}-{extra_id[:4]}')
    os.makedirs(output_directory)

    with open(os.path.join(output_directory, 'args.json'), 'w') as f:
        json.dump(vars(config), f)

    for file in tqdm(FILES):
        with open(os.path.join(output_directory, 'sample.json'), 'a') as write:
            with open(file, 'r') as f:
                for line in tqdm(f.readlines()):
                    e = json.loads(line)
                    prompt = e['summary']
                    for model in tqdm(ENGINES):
                        res = get_gpt_answer(prompt, engine=model)
                        e[model] = res
                    write.write(json.dumps(e) + '\n')
