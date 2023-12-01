import openai

'''
References for the prompt structure taken from: https://beta.openai.com/examples
'''
# Code explanation/summary
def explain_code(sample):
    prompt = f""" ### Explain the following code    
{sample}

### Here's what the above code is doing:
"""
    response = openai.Completion.create(model="code-davinci-002", prompt=prompt, temperature=0.7, best_of=10, n=3, max_tokens=max([len(sample)*2,1000]))    

# Simplify code 
def simplify_code(sample:str) -> str:
    prompt = f''' ### Restructure and simplify the following code without changing the logic
# Python 3.7
{sample}

# Simplified code
'''
    response = openai.Completion.create(model="code-davinci-002", prompt=prompt, temperature=0.7, best_of=10, n=3, max_tokens=max([len(sample)*2,1000]))

# Generate docstrings for functions
def generate_docstrings(sample:str)-> str:
    prompt = f''' ### Generate docstrings for the function below
# Python 3.7
{sample}

# An elaborate, high quality docstring for the above function:
'''
    response = openai.Completion.create(model="code-davinci-002", prompt=prompt, temperature=0.7, best_of=10, n=3, max_tokens=max([len(sample)*2.5,1000]))

#Language A to Python
def translate_code(sample:str,source_l,dest_l)-> str:
    prompt = f""" #Translate {source_l} to {dest_l} code translation:
# {source_l}
{sample}

# {dest_l}
"""
    response = openai.Completion.create(model="code-davinci-002", prompt=prompt, temperature=0.7, best_of=10, n=3, max_tokens=max([1000,len(sample)*2]))
    return response.choices[0].text

#fix bugs in the code
def fix_bugs(sample:str)-> str:
    prompt = f'''### Fix bugs in the function below
{sample}

### Fixed Python
'''
    response = openai.Completion.create(model="code-davinci-002", prompt=prompt, temperature=0.7, best_of=10, n=3, max_tokens=max([len(sample)*2,1000]))
    return response.choices[0].text

def trial(sample):
    prompt = f''' ### Fix bugs in the function below
{sample}

### Fixed Python
'''
    return prompt


