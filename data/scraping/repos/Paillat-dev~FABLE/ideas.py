import json
from utils.openaicaller import openai

with open('prompts/ideas.txt') as f:
    prompt = f.read()
    f.close()


async def generate_ideas(path, subject):
    prmpt = prompt.replace('[subject]', subject)
    try:
        with open(f'{path}/ideas.json', 'r') as f:
            ideas = f.read()
            ides_json = json.loads(ideas)
            f.close()
    except:
        ides_json = []
        ideas = "There are no existing ideas."
    exuisting_ideas = ""
    for idea in ides_json:
        exuisting_ideas += f"{idea['title']}\n"
    prmpt = prmpt.replace('[existing ideas]', exuisting_ideas)
    print(prmpt)
    response = await openai.generate_response(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"user","content":prmpt},
        ],
        )
    json_in_str= response['choices'][0]['message']['content'] # type: ignore
    json_obj = json.loads(json_in_str)
    for idea in json_obj:
        ides_json.append(idea)
    with open(f'{path}/ideas.json', 'w') as f:
        f.write(json.dumps(ides_json))
        f.close()