import argparse
from langchain.chat_models import ChatOpenAI
import re


def get_prompt(project_reqs: str, uml: str = None, structure: dict = None):
    base_prompt = f'''
    Generate code for the following project: 
    {project_reqs}
    {"UML: " + uml if uml else ""}
    {"Structure: " + structure.__str__() if structure else ""}

    Constraints:
    - Please only use javascript syntax and react-native API, don't use any other libraries
    - only reply code, don't reply any other text

    // Your javascript code here:
    '''
    return  base_prompt

llm = ChatOpenAI(temperature=0.9, model_name='gpt-3.5-turbo-0613')

def extract_code(string):
    pattern = r'```javascript\s+(.*?)\s+```'
    match = re.search(pattern, string, re.DOTALL)
    if match:
        code = match.group(1)
        return code
    else:
        return string

def write_code(project_reqs, uml, structure, out_file=None):
    prompt = get_prompt(project_reqs, uml, structure)
    res = llm.predict(prompt)
    extracted_code = extract_code(res)

    if out_file:
        with open(out_file, 'w') as f:
            f.write(extracted_code)

    return extracted_code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate code using OpenAI language model and optionally save it to a file.')
    parser.add_argument('--out_file', type=str, default=None, help='Output file path to save the generated code.')
    args = parser.parse_args()

    generated_code = write_code(out_file=args.out_file)

    if not args.out_file:
        print(generated_code)
