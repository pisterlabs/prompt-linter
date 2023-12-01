import os
import openai
from .utils import text2pdb

openai.api_key = os.getenv('OPENAI_API_KEY')

_vmd_cmd_prompt = '''> switch the representation to NewCartoon
changerep NewCartoon
> make them Ribbons
addrep $sel Ribbons
> switch style to CPK
changerep CPK
> enhance on it
AutoFocus $sel
> color by residue name
colorby ResName
> take a picture of this scene
Render
> get that part into focus
ZoomSel $sel
> rotate by 30 degrees along the y-axis
rotate y by 30 1
> Zoom in on them
ZoomSel $sel
'''
_vmd_select_prompt = '''> select the protein
set sel [atomselect top "protein"]
> select waters within five of residue number 10
set sel [atomselect top "water within 5 of resid 10"]
> select the glycines
set sel [atomselect top "resname GLY"]
> select the alpha carbons of residues lysines and valines
set sel [atomselect top "name CA and resname LYS VAL"]
> select the oxygen atoms in the lipids
set sel [atomselect top "name O and lipid"]
> select the fourth residue from the end
set sel [atomselect top "resid -4"]
'''


def _query_gpt3(query, training_string, T=0.20):
    prompt = '\n'.join([training_string, '> ' + query, ''])
    # return prompt
    response = openai.Completion.create(
        engine='davinci',
        prompt=prompt,
        temperature=T,
        max_tokens=64,
        top_p=1,
        best_of=4,
        frequency_penalty=0.0,
        presence_penalty=0,
        stop=['\n']
    )
    return response['choices'][0]['text'], response


def run_gpt_search(query):
    result = {'type': 'VMD Command'}
    if query.lower().find('select') > -1:
        # print(select_training_string)
        r, _ = _query_gpt3(query, _vmd_select_prompt)
        result['type'] = 'VMD Selection'
    elif query.lower().find('open') > -1:
        r = f'mol new {{{text2pdb(query)}}} type {{webpdb}} waitfor all;'
        result['type'] = 'PDB Search'
    else:
        r, _ = _query_gpt3(query, _vmd_cmd_prompt)
    result['data'] = r
    return result
