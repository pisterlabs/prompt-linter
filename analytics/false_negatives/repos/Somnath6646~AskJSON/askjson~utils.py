from openai import OpenAI
from termcolor import colored
import sys
import os
from io import StringIO
from dotenv import load_dotenv
import re
import subprocess
import sys
from stdlib_list import stdlib_list

STDLIB = set(stdlib_list())
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(
    api_key=OPENAI_API_KEY,
)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

def describe_json(json_obj, depth=0):
    indent = '  ' * depth
    description = ''

    if isinstance(json_obj, dict):
        description += "a JSON object which consists of the following keys and types:\n"
        common_keys = None

        if all(isinstance(val, dict) for val in json_obj.values()):
            common_keys = set(json_obj.values().__iter__().__next__().keys())
            for val in json_obj.values():
                common_keys &= set(val.keys())

        for key, value in json_obj.items():
            description += f"{indent}- {key}: {describe_json(value, depth+1)}"
            if common_keys and isinstance(value, dict):
                description += f"{indent}  All JSON objects in this level share the same keys: {', '.join(common_keys)}.\n"

    elif isinstance(json_obj, list):
        description += "an array of "
        if not json_obj:  # If the list is empty
            description += "no items\n"
        elif all(isinstance(item, dict) for item in json_obj):
            common_keys = set(json_obj[0].keys()) if json_obj else set()
            for item in json_obj:
                common_keys &= set(item.keys())

            if common_keys:
                description += f"JSON objects with a common structure. The structure is as follows:\n"
                description += describe_json(json_obj[0], depth+1)
            else:
                for index, item in enumerate(json_obj):
                    description += f"{indent}- Item {index+1}: {describe_json(item, depth+1)}"
        else:
            description += f"{describe_json(json_obj[0], depth+1)} (all elements in the array have this type)\n"

    else:
        description += f"a value of type {type(json_obj).__name__}\n"

    return description

def format_files(code):
    # Split the code into lines
    lines = code.split('\n')

    # Remove the first and last lines
    if len(lines) >= 2:
        del lines[0]
        del lines[-1]
    modified_code = '\n'.join(lines)

    return modified_code


def generate_response(prompt, model, max_tokens=1000):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
    )
    answer = ""
    res=""
    answer = ""
    
    for event in response:
      if "content" in res:
        print(colored(res["content"], "yellow"), end='', flush=True)   
        answer = answer+res["content"]
        
      event_text = dict(dict(dict(event)['choices'][0])['delta'])
      res = event_text

    
    return format_files(answer)


def execute_api_code(code):
    # Redirect stdout to capture print output
    original_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, globals())
        output = sys.stdout.getvalue()
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout

    return output





def get_imported_libraries(code):
    # Match 'import xyz' pattern
    direct_imports = re.findall(r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)', code, re.MULTILINE)
    
    # Match 'from xyz import ...' pattern
    from_imports = re.findall(r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import', code, re.MULTILINE)
    
    libraries = list(direct_imports) + list(from_imports)
    
    # Flatten the list and remove any spaces
    libraries = [lib.strip() for sublist in libraries for lib in sublist.split(',')]
    return libraries

def is_installed(library):
    if library in STDLIB:
        return True
    result = subprocess.run([sys.executable, "-m", "pip", "show", library], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # If the command was successful, the library is installed
    return result.returncode == 0


def install_library(library):
    subprocess.run([sys.executable, "-m", "pip", "install", library])

def install_missing_libraries(code):
    libraries = get_imported_libraries(code)
    print("\n\n")
    for library in libraries:
        if not is_installed(library):
           
            print(f"Installing {library}...")
            install_library(library)
            print(f"{library} installed!")
        else:
           
            print(colored(f"{library} is already installed.", "green"))




