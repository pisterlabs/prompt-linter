from prompts import Prompts
from open_ai_api import OpenAiApi

def format_and_get_response(prompt):
    prompts = Prompts(prompt)
    kg_prompt = prompts.kg_format()
    response = OpenAiApi().get_response(kg_prompt)
    return kg_prompt, response