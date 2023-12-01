import os
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tree_of_thoughts import OpenAILanguageModel, MonteCarloTreeofThoughts
from dotenv import load_dotenv

load_dotenv()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletion_with_backoff(**kwargs) -> openai.ChatCompletion:
    return openai.ChatCompletion.create(**kwargs)


def text_to_kg(prompt: str, text: str, model='gpt-3.5-turbo') -> str:
    """Function of Converting text to KG

    Args:
        prompt (str): instruction of the prompt
        text (str): input text
        model (str, optional): model name. Defaults to 'gpt-3.5-turbo'.

    Returns:
        str: knowledge graph
    """
    openai.api_key = os.getenv('OPENAI_API_KEY')

    messages = [{'role': 'user', 'content': prompt.replace('{prompt}', text)}]
    completion = chatcompletion_with_backoff(model=model,
                                             max_tokens=300,
                                             temperature=0,
                                             messages=messages)

    res = completion.choices[0].message.content
    return res


def text_to_kg_with_tot(prompt: str,
                        text: str,
                        api_model='gpt-3.5-turbo') -> str:
    openai.api_key = os.getenv('OPENAI_API_KEY')

    model = OpenAILanguageModel(api_key=os.getenv('OPENAI_API_KEY'), api_model=api_model)
    tree_of_thoughts = MonteCarloTreeofThoughts(model)

    num_thoughts = 1
    max_steps = 3
    max_states = 4
    pruning_threshold = 0.5

    solution = tree_of_thoughts.solve(
        initial_prompt=prompt.replace('{prompt}', text),
        num_thoughts=num_thoughts,
        max_steps=max_steps,
        max_states=max_states,
        pruning_threshold=pruning_threshold,
        # sleep_time=sleep_time
    )

    print(f"Solution: {solution}")

    return solution


if __name__ == '__main__':
    import time
    import pandas as pd
    from pprint import pprint

    with open('src/prompts/text_to_kg.prompt', 'r') as f:
        prompt = f.read()

    data = pd.read_json('src/data/KQAPro/train.json')
    data = data.iloc[:10]
    kg_triplets = []

    model = 'gpt-4'
    for i, item in data.iterrows():
        msg = item['question']

        print(i, '=' * 30)
        pprint(msg)
        print('->')

        res = text_to_kg_with_tot(prompt, msg, model)
        kg_triplets.append(str(res))

        pprint(res)
        print()
        time.sleep(0.2)

    data['kg_triplets'] = kg_triplets
    data.to_json(f'src/output/KQAPro_train_kg_{model}_tot.json', orient='records')