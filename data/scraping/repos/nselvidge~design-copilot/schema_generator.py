import json
import os
import zipfile
from io import BytesIO
from urllib.parse import urlparse
from urllib.request import urlopen
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import base64
from dotenv import load_dotenv
from ghapi.all import GhApi

load_dotenv('../.env')
GH_TOKEN = os.getenv("GH_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

example_schema = {
  "Article": {
    "description": "The Article model represents a blog post authored by a User, which can have comments and favorites, features certain field validations, generates a unique slug, and supports tagging and user-specific queries.",
    "attributes": {
      "id": "integer",
      "title": "string",
      "body": "text",
      "slug": "string",
      "created_at": "datetime",
      "updated_at": "datetime",
      "user_id": "integer"
    },
    "associations": {
      "belongs_to": ["User"],
      "has_many": ["Favorite", "Comment", "Article"]
    },
    "scopes": {
      "authored_by": "Returns all articles authored by a particular user",
      "favorited_by": "Returns all articles favorited by a particular user"
    },
    "tags": "Supported, by 'acts-as-taggable-on' gem"
  }
}


def zipfile_from_github(repo_url, main_branch="master"):
    folder_prefix, zip_url = compute_prefix_and_zip_url(repo_url, main_branch)
    http_response = urlopen(zip_url)
    zf = BytesIO(http_response.read())
    return zipfile.ZipFile(zf, "r"), folder_prefix


def compute_prefix_and_zip_url(repo_url, main_branch="master"):
    parsed = urlparse(repo_url)
    if not all([parsed.scheme, parsed.netloc]):
        raise ValueError("Invalid URL: " + repo_url)

    path_parts = parsed.path.strip("/").split("/")
    repo_name = path_parts[-1]
    if not repo_name:
        raise ValueError("Invalid repository URL: " + repo_url)

    folder_prefix = f"{repo_name}-{main_branch}"

    # Ensure that the URL is a GitHub repository URL
    if parsed.netloc != "github.com":
        raise ValueError("Invalid GitHub repository URL")

    # Extract the username and repository name
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub repository URL")

    username = path_parts[0]
    repo = path_parts[1]

    # Construct the .zip file URL
    zip_url = (
        f"https://github.com/{username}/{repo}/archive/refs/heads/{main_branch}.zip"
    )

    return folder_prefix, zip_url


def list_files_in_models_folder(repo_url, branch_name):
    zip_file, folder_prefix = zipfile_from_github(repo_url, branch_name)

    print("it works.")
    model_files = []

    for file in zip_file.namelist():
        # Check if the file is directly in the "models" folder
        if (file.startswith(f"{folder_prefix}/app/models")
                and not file.endswith('/')
                and not os.path.basename(file).startswith('.')
                and file.count('/') == 3):  # Do not read subdirectories
            print(file)
            model_files.append(file)

    return model_files


def get_file_content(repo_url, branch_name, file_path):
    api = GhApi(token=GH_TOKEN)

    path_parts = urlparse(repo_url).path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub repository URL")

    username = path_parts[0]
    repo = path_parts[1]

    # This will remove 'rails-realworld-example-app-master/' from the file_path
    relative_file_path = file_path.replace(f"{repo}-{branch_name}/", "")

    try:
        commits = api.repos.list_commits(username, repo, path=relative_file_path)
    except Exception as e:
        print(f"Failed to get commits for file: {relative_file_path}")
        print(f"Error: {e}")
        return None

    if commits:
        latest_commit_sha = commits[0].sha
        try:
            file_content = api.repos.get_content(username, repo, path=relative_file_path, ref=latest_commit_sha)
            decoded_content = base64.b64decode(file_content.content).decode("utf-8")
            num_lines = len(decoded_content.splitlines())

            # Check if the file is too large
            if num_lines > 10000:
                print(f"File {relative_file_path} is too large, skipping...")
                return None

            return decoded_content
        except Exception as e:
            print(f"Failed to get file content for file: {relative_file_path}")
            print(f"Error: {e}")
    else:
        print(f"No commits found for file: {relative_file_path}")

    return None


def read_prompt_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
        return content


def generate_json_from_models(repo_url, branch_name, model_files, uuid):
    json_model_dict = {}
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-16k-0613", temperature=0.2
    )
    for file in model_files:
        file_content = get_file_content(repo_url, branch_name, file)
        # Here we use the function read_prompt_from_file to get the prompt template
        prompt_template = read_prompt_from_file("prompts/json_generator.txt")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["file_content", "example_schema"]
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        output = llm_chain.run({'file_content': file_content, 'example_schema': example_schema})
        # Convert the output from string format to JSON object
        json_output = json.loads(output)

        # Get the class name from the output JSON
        class_name = list(json_output.keys())[0]

        json_model_dict[class_name] = json_output[class_name]

    # Save json_model_dict to a JSON file
    with open(f'{uuid}.json', 'w') as json_file:
        json.dump(json_model_dict, json_file, indent=4)

    return json_model_dict


if __name__ == "__main__":
    repo_url = "https://github.com/discourse/discourse"
    branch_name = "main"
    model_files = list_files_in_models_folder(repo_url, branch_name)
    print(model_files)
    if model_files:
        generate_json_from_models(repo_url, branch_name, model_files)
