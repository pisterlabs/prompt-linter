import subprocess
import argparse
import sqlite3
import logging
import os

def create_directory(parent_dir, dir_name=None):
    if dir_name:
        dir_name = os.path.join(parent_dir, dir_name)
    else:
        dir_name = parent_dir

    logging.info(f"Creating directory {dir_name}")
    if os.path.exists(dir_name):
        logging.error(f"Directory {dir_name} already exists.")
        return False

    try:
        os.makedirs(dir_name, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create directory {dir_name}: {e}")
        return False

    return True

def create_ron_testing_dir():
    logging.info("Creating RONTESTING directory")
    os.makedirs("RONTESTING", exist_ok=True)

def create_workspace_dir():
    logging.info("Creating WORKSPACE directory")
    os.makedirs("WORKSPACE", exist_ok=True)

def create_sqlite_db(db_name):
    conn = sqlite3.connect(f"{db_name}")
    logging.info(f"Created SQLite database {db_name}")
    conn.close()

def create_readme():
    with open("README.md", "w") as f:
        f.write("Here are some uselful commands you might find useful:\n")
        f.write("\n")   ###fill this in with poetry commands
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")

def print_example_usage():
    print("Example usage: python3 pybuild.py --dir my_project --packages numpy,pandas,matplotlib --python 3.11.3\n")

def create_openai_script():
    content = '''
import openai
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv("../.env")

# Get API Key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API Key
openai.api_key = OPENAI_API_KEY

# Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Translate the following English text to French: 'Hello, how are you?'"}
]

# Generate response
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=60
)

print(response['choices'][0]['message']['content'])
'''
    with open("app/openai_script.py", "w") as f:
        f.write(content.strip())

def create_env_file():
    default_env_vars = [
        "#openai",
        "OPENAI_API_KEY=",
        "#aws",
        "S3_BUCKET=",
        "AWS_ACCESS_KEY_ID=",
        "AWS_SECRET_ACCESS_KEY=",
        "AWS_REGION=",
        "#django",
        "DJANGO_SECRET_KEY=",
        "#pinecone",
        "DATASTORE=pinecone",
        "PINECONE_API_KEY=",
        "PINECONE_ENVIRONMENT=",
        "PINECONE_INDEX="
    ]
    with open('.env', 'w') as f:
        f.write('\n'.join(default_env_vars))

def create_and_populate_poetry_file():
    # Create a basic pyproject.toml file
    content = '''
    [tool.poetry]
    name = "gpt-buildbox"
    version = "1.4.2"
    description = "A GPT-powered Python sandbox."
    authors = ["Your Name <youremail@example.com>"]

    [tool.poetry.dependencies]
    python = "^3.8"

    [build-system]
    requires = ["poetry-core>=1.4.2"]
    build-backend = "poetry.core.masonry.api"
    '''

    with open("pyproject.toml", "w") as f:
        f.write(content)

    # Add dependencies with poetry
    subprocess.run(["poetry", "add", "openai", "python-dotenv"], check=True)

def create_gitignore_file():
    default_ignore_files = [ 
        "database.sqlite3",
        "RONTESTING/", 
        ".env", 
        "__pycache__/",  # Ignore Python cache files
        "*.pyc",  # Ignore Python compiled files
        "*.pyo",
        "*.pyd",
        ".Python",
        "ipynb_checkpoints/",  # Ignore Jupyter Notebook checkpoints
        ".vscode/",  # Ignore VS Code settings
        "*.log",  # Ignore all log files
        ".DS_Store",  # Ignore Mac system files
        "dist/",  # Ignore Python distribution folder
        "build/",  # Ignore Python build folder
        "*.egg-info/",  # Ignore Python egg info
        ".mypy_cache/",  # Ignore mypy type checker cache
    ]
    with open('.gitignore', 'w') as f:
        f.write('\n'.join(default_ignore_files))

def initialize_git_repo():
    logging.info("Initializing Git repository")
    subprocess.call(['git', 'init'])
    subprocess.call(['git', 'checkout', '-b', 'main'])  # switch to a new branch 'main'
    subprocess.call(['git', 'add', '.'])
    subprocess.call(['git', 'commit', '-m', 'Initial commit'])

def print_activate_virtual_env_command():
    print(f"To activate the virtual environment, run: poetry shell\n")
    print(f"poetry shell\n")

def run_tree():
    subprocess.call(['tree', '-I', '.DS_Store'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Python project")
    parser.add_argument("--dir", help="Directory name", default="project")
    parser.add_argument("--packages", help="Packages to install (separated by commas)", default="")
    parser.add_argument("--python", help="Python version", default="3.11.3")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not create_directory(args.dir):
        logging.error("Failed to create directory")
        exit(1)

    os.chdir(args.dir)
    create_directory(".", "app")
    create_sqlite_db('database.sqlite3')
    create_ron_testing_dir()
    create_workspace_dir()
    create_readme()
    create_openai_script()
    create_env_file()
    create_and_populate_poetry_file()
    create_gitignore_file()
    initialize_git_repo()
    run_tree()
    print_activate_virtual_env_command()
    print_example_usage()