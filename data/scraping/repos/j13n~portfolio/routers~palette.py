from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse

import os
import json
import openai

openai.api_key = os.environ.get('OPENAI_API_KEY')

router = APIRouter()


def complete_text(sentence):
    prompt = f"""
    You are a color palette generating assistant that responds to text prompts for color palettes
    Your should generate color palettes that fit the theme, mood, or instructions in the prompt.
    The palettes should be between 2 and 8 colors.

    Q: The Mediterranean Sea
    A: ["#006699", "#66CCCC", "#F0E68C", "#008000", "#F08080"]

    Q: sage, nature, earth
    A: ["#EDF1D6", "#9DC08B", "#609966", "#40513B"]

    Q: The colors of the Dutch flag
    A: ["#21468B", "#FFFFFF", "#AE1C28"]

    Desired Format: a JSON array of hexadecimal color codes

    Q: Convert the following verbal description of a color palette into a list of colors: {sentence} 
    A:
    """

    completion = openai.Completion.create(
        prompt=prompt, model="text-davinci-003", max_tokens=200
    )

    return json.loads(completion.choices[0].text)


@router.post('/generate')
async def generate(prompt=Form('prompt')):
    divs = ""
    for color in complete_text(prompt):
        divs += f'<div class="bg-[{color}] hover:mix-blend-difference relative flex justify-center"><div class="absolute">{color}</div></div>'
    return HTMLResponse(divs)
