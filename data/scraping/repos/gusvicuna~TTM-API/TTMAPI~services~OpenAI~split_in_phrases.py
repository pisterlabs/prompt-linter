import json
import openai
from dotenv import dotenv_values

config = dotenv_values("settings.env")

openai.api_key = config["OPENAI_API_KEY"]

prompt_instruction = "Separa el siguiente texto en frases." +\
    "Debes entregarme el resultado en una lista de strings, cada uno siendo una frase.\n" +\
    "Ejemplo:" + "Input: La vida es grande y la muerte también aunque ambas no existen" +\
    'Output: ["La vida es grande", "y la muerte también", "aunque ambas no existen"].'


def split_in_phrases(text: str, logger):
    response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt_instruction},
                {"role": "user", "content": text}],
            temperature=0.2,
            max_tokens=80,
            top_p=0.4,
            frequency_penalty=0,
            presence_penalty=0
        )
    result = response['choices'][0]['message']['content']
    logger.debug(f"result: {result}")
    return json.loads(result)
