"""Personalized AI interaction to provide health advice."""

from openai import OpenAI
import time
from service.properties import ApiKeySelector, load_prompts_map
from utils import timer
from typing import Dict, List
from retrying import retry
from string import Template

API_SELECTOR = ApiKeySelector()
PROMPTS_MAP = load_prompts_map()


class AIAdvisor:
    """
    Provide personalized health advice from GPT
    """

    @timer
    def get_body_index_advice(self, request_data: Dict, use_gpt4: bool = False) -> Dict:
        print(f'Generating AI personalized advice now for member based on body index')
        result_body_index_json = {}
        try:
            gpt_model = 'gpt-4' if use_gpt4 else 'gpt-3.5-turbo'
            result_body_index_json = generate_member_body_index_advice(PROMPTS_MAP['user_prompt'],
                                                                       PROMPTS_MAP['system_prompt'],
                                                                       request_data,
                                                                       gpt_model)
        except Exception as e:
            print(f"Met exception {e} during AI advice generation for body index analysis")
        return result_body_index_json


@timer
@retry(stop_max_attempt_number=1, wait_fixed=1000, retry_on_result=lambda result: result is None)
def generate_member_body_index_advice(user_prompt_tpl, sys_prompt, input_params: Dict, model) -> List or None:
    """Invoke GPTs to generate AI health advice based on body index data."""
    ai_response_advice = None
    try:
        user_prompt = Template(user_prompt_tpl).substitute(**input_params)
        ai_response_advice = invoke_gpt_api(sys_prompt, user_prompt,
                                            model=model, max_tokens=2048)
    except Exception as e:
        print(f"Met exception when generating AI health advice, exception msg {e}, retrying...")
    finally:
        return ai_response_advice


def invoke_gpt_api(system_prompt: str,
                   user_prompt: str,
                   model: str = 'gpt-3.5-turbo',
                   max_tokens: int = 2048,
                   temperature: float = 0.7,
                   frequency_penalty: float = 0.25):
    if model == 'gpt-4':
        # gpt-4 api key
        selected_api_key = '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'
    else:
        # gpt-3.5-turbo key
        selected_api_key = API_SELECTOR.retrieve_api_key()
    print("currently used api key is: ", selected_api_key)
    messages = [{
        "role": "system",
        "content": system_prompt.strip()
    }, {
        "role": "user",
        "content": user_prompt.strip()
    }]
    try:
        client = OpenAI(
            api_key=selected_api_key
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=frequency_penalty,
            presence_penalty=0.0,
        )
        assistant_response = response.choices[0].message.content
        return assistant_response.strip()
    except Exception as e:
        print(f"Exception met when invoking openai api with api key {selected_api_key}, error msg: {e}")
        return ""


if __name__ == '__main__':
    # start_time0 = time.time()
    input_case1 = {
        "name": "Ethan",
        "track_data": [
            {"date": "2023-10-01", "height": "175cm", "weight": "68kg", "body_fat_percentage": 0.18},
            {"date": "2023-10-08", "height": "175cm", "weight": "66kg", "body_fat_percentage": 0.17},
            {"date": "2023-10-15", "height": "175cm", "weight": "64kg", "body_fat_percentage": 0.16},
            {"date": "2023-11-01", "height": "175cm", "weight": "60kg", "body_fat_percentage": 0.14}
        ]
    }
    advisor = AIAdvisor()
    test_result1 = advisor.get_body_index_advice(input_case1)

    print(test_result1)
