import openai
import pprint
import os


openai.organisation = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")

GPT4 = 'gpt-4-0314'
MODEL_NAME = GPT4
model = openai.Model(MODEL_NAME)

def list_all_models():
    model_list = openai.Model.list()['data']
    model_ids = [x['id'] for x in model_list]
    model_ids.sort()
    pprint.pprint(model_ids)

if __name__ == '__main__':
    list_all_models()