import os
import openai

def get_gpt_match(prompt, key, model="text-davinci-002"):
    # mykey = b'Z1QFxceGL_s6karbgfNFyuOdQ__m5TfHR7kuLPJChgs='
    # enc = b'gAAAAABjRh0iNbsVb6_DKSHPmlg3jc4svMDEmKuYd-DcoTxEbESYI9F8tm8anjbsTsZYHz_avZudJDBdOXSHYZqKmhdoBcJd919hCffSMg6WFYP12hpvI7EeNppGFNoZsLGnDM5d6AOUeRVeIc2FbmB_j0vvcIwuEQ=='
    # fernet = Fernet(mykey)
    # openai.api_key = fernet.decrypt(enc).decode()
    openai.api_key = key
    response = openai.Completion.create(model=model, prompt=prompt, temperature=0.0, max_tokens=1024)
    result = response.choices[0].text.strip()
    # print(result)
    return result

# Get gpt-3 prompt with arizona-extraction, ontology terms and match targets
def get_code_dkg_prompt(vars, terms, target):
    text_file = open("prompts/prompt.txt", "r")
    prompt = text_file.read()
    text_file.close()

    vstr = ''
    vlen = len(vars)
    i = 1
    for v in vars:
        vstr += str(i) + " (" + str(v[1]) + ", " + str(v[2]) + ")\n"
        i += 1
    # print(vstr)
    tstr = '[' + ', '.join(terms) + ']'
    tlen = len(terms)
    # print(tstr)
    prompt = prompt.replace("[VAR]", vstr)
    prompt = prompt.replace("[VLEN]", str(vlen))
    prompt = prompt.replace("[TERMS]", tstr)
    prompt = prompt.replace("[TLEN]", str(tlen))
    prompt = prompt.replace("[TARGET]", target)
    # print(prompt)
    return prompt


# Get gpt-3 prompt with formula, code terms and match formula targets
def get_code_formula_prompt(code, formula, target):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/code_formula_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[CODE]", code)
    prompt = prompt.replace("[FORMULA]", formula)
    prompt = prompt.replace("[TARGET]", target)
    # print(prompt)
    return prompt

def get_petri_places_prompt(text):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/petri_places_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[TEXT]", text)
    return prompt

def get_petri_match_place_prompt(text, place):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/petri_match_place_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[TEXT]", text)
    prompt = prompt.replace("[PLACE]", place)
    return prompt

def get_petri_match_dataset_prompt(place, text, columns):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/petri_match_dataset_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[PLACE]", place)
    prompt = prompt.replace("[TEXT]", text)
    prompt = prompt.replace("[COLUMNS]", columns)

    return prompt

def get_petri_init_param_prompt(text, param):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/petri_init_param_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[TEXT]", text)
    prompt = prompt.replace("[PARAM]", param)
    return prompt

def get_petri_transitions_prompt(text):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/petri_transitions_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[TEXT]", text)
    return prompt

def get_petri_arcs_prompt(text):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/petri_arcs_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[TEXT]", text)
    return prompt


# Get gpt-3 prompt with formula, code terms and match formula targets
def get_code_text_prompt(code, text, target):
    text_file = open(os.path.join(os.path.dirname(__file__), 'prompts/code_text_prompt.txt'), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[CODE]", code)
    prompt = prompt.replace("[TEXT]", text)
    prompt = prompt.replace("[TARGET]", target)
    # print(prompt)
    return prompt


# Get gpt-3 prompt with code, dataset and match function targets
def get_code_dataset_prompt(code, dataset, target):
    text_file = open(os.path.join(os.path.dirname(__file__), "prompts/code_dataset_prompt.txt"), "r")
    prompt = text_file.read()
    text_file.close()

    prompt = prompt.replace("[CODE]", code)
    prompt = prompt.replace("[DATASET]", dataset)
    prompt = prompt.replace("[TARGET]", target)
    # print(prompt)
    return prompt



