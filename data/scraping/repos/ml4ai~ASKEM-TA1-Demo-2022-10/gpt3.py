import re
from cryptography.fernet import Fernet
import openai


def get_gpt_match(prompt):
    mykey = b'Z1QFxceGL_s6karbgfNFyuOdQ__m5TfHR7kuLPJChgs='
    enc = b'gAAAAABjRh0iNbsVb6_DKSHPmlg3jc4svMDEmKuYd-DcoTxEbESYI9F8tm8anjbsTsZYHz_avZudJDBdOXSHYZqKmhdoBcJd919hCffSMg6WFYP12hpvI7EeNppGFNoZsLGnDM5d6AOUeRVeIc2FbmB_j0vvcIwuEQ=='
    fernet = Fernet(mykey)
    openai.api_key = fernet.decrypt(enc).decode()
    # prompt = "Please write me a sentence\n\n"
    response = openai.Completion.create(model="text-davinci-002", prompt=prompt, temperature=0.0, max_tokens=256)
    result = response.choices[0].text.strip()
    # print(result)
    return result

def get_prompt(vars, terms, target):
    text_file = open("prompt.txt", "r")
    prompt = text_file.read()
    text_file.close()

    vstr = ''
    vlen = len(vars)
    i = 1;
    for v in vars:
        vstr += str(i) + " (" + v[1]+", "+v[2]+")\n"
        i+=1;
    # print(vstr)
    tstr = '['+', '.join(terms)+']'
    tlen = len(terms)
    # print(tstr)
    prompt = prompt.replace("[VAR]", vstr)
    prompt = prompt.replace("[VLEN]", str(vlen))
    prompt = prompt.replace("[TERMS]", tstr)
    prompt = prompt.replace("[TLEN]", str(tlen))
    prompt = prompt.replace("[TARGET]", target)
    # print(prompt)
    return prompt

def get_variables(path):
    list = []
    with open(path) as myFile:
        for num, line in enumerate(myFile, 1):
            match = re.match(r'\s*(\S+)\s*=\s*([-+]?(?:\d*\.\d+|\d+))\s*', line)
            if match:
                para = match.group(1)
                val = match.group(2)
                # print(num, ",", para, ",", val)
                list.append((num, para, val))
    # print(len(list))
    return list

def get_match(vars, terms, target):
    prompt = get_prompt(vars, terms, target)
    match = get_gpt_match(prompt)
    val = match.split("(")[1].split(",")[0]
    return val

def match_targets(targets, code_path, terms):
    vars = get_variables(code_path)
    vdict = {}
    connection = [];
    for idx, v in enumerate(vars):
        vdict[v[1]] = idx
    for t in targets:
        val = get_match(vars, terms, t)
        connection.append((t,{val: "grometSubObject"}, float(vars[vdict[val]][2]),  vars[vdict[val]][0]))
    return connection

# terms = ['population', 'doubling time', 'recovery time', 'infectious time']
# code = "CHIME_SIR_default_model.py"
# targets = ['population', 'doubling time', 'recovery time', 'infectious time']
# val = match_targets(targets, code, terms)
# print(val)
