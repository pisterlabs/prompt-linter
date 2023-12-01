import openai
from dotenv import dotenv_values

config = dotenv_values("settings.env")

openai.api_key = config["OPENAI_API_KEY"]

prompt_instruction = "Corrige la ortografía, gramática y puntuación."


def fix_grammar(traintext: str):
    response = openai.Edit.create(
        model="text-davinci-edit-001",
        input=traintext,
        instruction=prompt_instruction,
        temperature=0.5,
        top_p=0.3
        )
    return response["choices"][0]["text"]
