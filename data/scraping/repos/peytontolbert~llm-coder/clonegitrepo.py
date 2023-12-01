import os
import subprocess
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from utils import clean_dir
import shutil
from glob import glob
from openai.embeddings_utils import get_embedding
import pandas as pd
import openai


from dotenv import load_dotenv
# Initialize OpenAI and GitHub API keys
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_file_paths(repo_url, clone_dir):
    # Clone the repo
    subprocess.run(['git', 'clone', repo_url, clone_dir])
    
    # Walk the directory and get all file paths
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(clone_dir):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))
    
    return file_paths

def get_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        print(f'Skipped file due to ecoding issues: {file_path}')
        return None

def get_function_name(code):
    """
    Extract function name from a line beginning with "def "
    """
    assert code.startswith("def ")
    return code[len("def "): code.index("(")]


def get_until_no_space(all_lines, i) -> str:
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, i + 10000):
        if j < len(all_lines):
            if len(all_lines[j]) == 0 or all_lines[j][0] in [" ", "\t", ")"]:
                ret.append(all_lines[j])
            else:
                break
    return "\n".join(ret)


def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    whole_code = open(filepath).read().replace("\r", "\n")
    all_lines = whole_code.split("\n")
    for i, l in enumerate(all_lines):
        if l.startswith("def "):
            code = get_until_no_space(all_lines, i)
            function_name = get_function_name(code)
            yield {"code": code, "function_name": function_name, "filepath": filepath}



def main():
    load_dotenv()
    repo_url = os.getenv('REPO_URL')
    clone_dir = os.getenv('CLONE_DIR')
    file_paths = get_file_paths(repo_url, clone_dir)
    print(*file_paths, sep='\n')
    code_files = [y for x in os.walk(clone_dir) for y in glob(os.path.join(x[0], '*.py'))]
    print("Total number of py files:", len(code_files))
    if len(code_files) == 0:
        print("Double check that you have downloaded the repo and set the code_dir variable correctly.")

    all_funcs = []
    for code_file in code_files:
        funcs = list(get_functions(code_file))
        for func in funcs:
            all_funcs.append(func)
    print("Total number of functions:", len(all_funcs))
    df = pd.DataFrame(all_funcs)
    df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, engine="text-embedding-ada-002")) 
    df['filepath'] = df['filepath'].apply(lambda x: x.replace(clone_dir, ""))
    df.to_csv("functions2.csv", index=True)
    df.head()


if __name__ == "__main__":
    main()