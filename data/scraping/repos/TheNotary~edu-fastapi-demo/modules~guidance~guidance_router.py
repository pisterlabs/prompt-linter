from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from helpers.jinja_helpers import build_templates_and_router
import guidance
import pdb

templates, router, module_name = build_templates_and_router(__file__)



# examples = [
#     {'input': 'I wrote about shakespeare',
#     'entities': [{'entity': 'I', 'time': 'present'}, {'entity': 'Shakespeare', 'time': '16th century'}],
#     'reasoning': 'I can write about Shakespeare because he lived in the past with respect to me.',
#     'answer': 'No'},
#     {'input': 'Shakespeare wrote about me',
#     'entities': [{'entity': 'Shakespeare', 'time': '16th century'}, {'entity': 'I', 'time': 'present'}],
#     'reasoning': 'Shakespeare cannot have written about me, because he died before I was born',
#     'answer': 'Yes'}
# ]


# {{~! place the real question at the end }}
# define the guidance program



class InputData(BaseModel):
    input_data: str = Field(..., min_length=1, description="Input data must be at least 1 character long")

@router.get("")
async def guidance(request: Request):
    return templates.TemplateResponse("modules/" + module_name + "/index.html", {"request": request})

@router.post("")
async def guidance(json: InputData):
    user_input = json.input_data

#     structure_program = guidance(
# '''// The below code will
# // define a javascript function that plots the line for a graph in a canvas element.
# // An example input would be "y = 2x + 3" and the output should be a graph with the equation plotted on it.
# function plot(equationForLine) {
#   {{~gen 'code'}}
# }
# ''')

    guidance.llm = guidance.llms.Transformers("bigcode/starcoderplus", device=None)
    # guidance.llm = guidance.llms.Transformers("TheBloke/Kimiko-7B-fp16", device=0)


    structure_program = await guidance(user_input)

    pdb.set_trace()

    # messages = [{ "role": "user", "content": user_input }]
    # resp = gptj.chat_completion(messages)

    # execute the program
    out = structure_program(
        input='The T-rex bit my dog'
    )

    pdb.set_trace()

    return "done"

    # return resp['choices'][0]['message']['content']
