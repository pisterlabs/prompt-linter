import base64
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from embeddings import metadata_field_info, vector_store
from ghapi.all import GhApi
from git import create_github_pull_request
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SelfQueryRetriever

from convex import ConvexClient

# get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# go up one level to get the root directory
root_dir = os.path.dirname(current_dir)

dotenv_path = os.path.join(root_dir, ".env.local")

# load the .env file
load_dotenv(dotenv_path)

client = ConvexClient(os.getenv("NEXT_PUBLIC_CONVEX_URL"))

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY", "")
GH_TOKEN = os.getenv("GH_TOKEN", "")
RAJA_TOKEN = os.getenv("GH_TOKEN_RAJA", "")
LLM = ChatOpenAI(
    openai_api_key=OPEN_AI_KEY, model_name="gpt-3.5-turbo-16k-0613", temperature=0.2
)


def load_template_info():
    template_variables = {
        "name",
        "label",
        "description",
        "acceptance_criteria",
        "how_to_reproduce",
        "current_filepath",
        "file_objects",
    }

    branch_name_variables = {"name", "label"}
    pr_variables = {
        "name",
        "label",
        "description",
        "how_to_reproduce",
        "acceptance_criteria",
    }

    branch_name_filepath = "prompts/pr_branch_template.txt"
    pr_title_filepath = "prompts/pr_title_template.txt"
    pr_body_filepath = "prompts/pr_body_template.txt"
    pr_commit_message_filepath = "prompts/pr_commit_message_template.txt"

    return (
        template_variables,
        branch_name_variables,
        pr_variables,
        branch_name_filepath,
        pr_title_filepath,
        pr_body_filepath,
        pr_commit_message_filepath,
    )


# Get template info
(
    template_variables,
    branch_name_variables,
    pr_variables,
    branch_name_filepath,
    pr_title_filepath,
    pr_body_filepath,
    pr_commit_message_filepath,
) = load_template_info()

TEMPLATE_VARIABLES = {
    "bug": {
        "bug": template_variables,
        "branch_name": branch_name_variables,
        "pr_title": pr_variables,
        "pr_body": pr_variables,
        "pr_commit_message": pr_variables,
    },
    "feature": {
        "feature": template_variables,
        "branch_name": branch_name_variables,
        "pr_title": pr_variables,
        "pr_body": pr_variables,
        "pr_commit_message": pr_variables,
    },
}

TEMPLATE_FILEPATHS = {
    "bug": {
        "bug": "prompts/tickets/bug_template.txt",
        "branch_name": branch_name_filepath,
        "pr_title": pr_title_filepath,
        "pr_body": pr_body_filepath,
        "pr_commit_message": pr_commit_message_filepath,
    },
    "feature": {
        "feature": "prompts/tickets/feature_template.txt",
        "branch_name": branch_name_filepath,
        "pr_title": pr_title_filepath,
        "pr_body": pr_body_filepath,
        "pr_commit_message": pr_commit_message_filepath,
    },
}


@dataclass
class Card:
    type: str
    name: str = ""
    label: str = ""
    description: str = ""
    acceptance_criteria: str = ""
    how_to_reproduce: str = ""
    current_filepath: str = ""
    file_objects: str = ""
    status: str = "Ready to Deploy"

    def __post_init__(self):
        self.type = self.type.lower()
        self.name = self.name.lower()
        self.label = self.label.lower()
        self.description = self.description.lower()
        self.acceptance_criteria = self.acceptance_criteria.lower()
        self.how_to_reproduce = self.how_to_reproduce.lower()


def load_template_from_file(filepath, variables):
    try:
        with open(filepath, "r") as file:
            template_str = file.read()

        # Find all placeholders in the template string
        placeholders = re.findall(r"\{(\w+)\}", template_str)

        # Create a dictionary containing only keys that exist in the template string
        filtered_variables = {
            key: "{" + key + "}" for key in placeholders if key in variables
        }

        # Format the template string using the filtered dictionary
        return template_str.format(**filtered_variables)

    except Exception as e:
        print(f"Failed to load template from file: {filepath}")
        print(f"Error: {e}")
        raise


def load_prompt_from_file(filepath, variables):
    try:
        with open(filepath, "r") as file:
            template_str = file.read()
        return template_str.format(**variables)
    except Exception as e:
        print(f"Failed to load template from file: {filepath}")
        print(f"Error: {e}")
        raise


def run_llm_chain(template_str, **kwargs):
    # Find all placeholders in the template string
    placeholders = re.findall(r"\{(\w+)\}", template_str)

    # Filter kwargs to only include keys that exist in the template string
    filtered_kwargs = {key: kwargs[key] for key in placeholders if key in kwargs}

    prompt = PromptTemplate(
        template=template_str, input_variables=list(filtered_kwargs.keys())
    )
    llm_chain = LLMChain(prompt=prompt, llm=LLM)
    return llm_chain.run(**filtered_kwargs)


def generate_code(file, ticket_type, template_name, variables, documents, **kwargs):
    for document in documents:
        filepath = document.metadata["document_id"]
        print("Generating code for document:", filepath)
        if not filepath.endswith((".py", ".ts", ".tsx", ".js", ".jsx")):
            print("Skipping file:", filepath)
            continue
        filepath = filepath.split("/")
        filepath = "/".join(filepath[1:])
        print(filepath)
        kwargs["current_filepath"] = filepath
        template_str = load_template_from_file(
            TEMPLATE_FILEPATHS[ticket_type][template_name], variables
        )
        file[filepath] = run_llm_chain(template_str, **kwargs)
    return file


def raja_agent(req_body):
    card = Card(**req_body)
    file, metadata = {}, {}

    document_description = "Stores the code in the file"

    retriever = SelfQueryRetriever.from_llm(
        LLM, vector_store, document_description, metadata_field_info, verbose=True
    )

    # Determine the appropriate file path based on the card type
    if card.type == "bug":
        get_relevant_file_path = "prompts/get_relevant_files_bug.txt"
    elif card.type == "feature":
        get_relevant_file_path = "prompts/get_relevant_files_feature.txt"
    else:
        raise ValueError(f"Unsupported card type: {card.type}")

    get_relevant_file_paths = load_prompt_from_file(get_relevant_file_path, vars(card))

    print(get_relevant_file_paths)

    relevant_documents = retriever.get_relevant_documents(get_relevant_file_paths)

    for template_name, variables in TEMPLATE_VARIABLES[card.type].items():
        kwargs = {var: getattr(card, var, "") for var in variables}

        if template_name == card.type:
            file_objects = ""
            for document in relevant_documents:
                repo_info = client.query("repo:get")[0]
                repo_name = repo_info["name"]
                repo_owner = repo_info["owner"]
                ghapi_client = GhApi(owner=repo_owner, repo=repo_name, token=GH_TOKEN)
                ghapi_raja_client = GhApi(
                    owner=repo_owner, repo=repo_name, token=RAJA_TOKEN
                )
                file_path = document.metadata["document_id"]
                print(file_path)

                # Remove repository name from the file path if it is there
                repo_name_with_slash = f"{repo_name}-main/"
                truncated_file_path = file_path
                if file_path.startswith(repo_name_with_slash):
                    truncated_file_path = file_path.replace(repo_name_with_slash, "", 1)

                try:
                    commits = ghapi_client.repos.list_commits(path=truncated_file_path)
                except Exception as e:
                    print(f"Failed to get commits for file: {file_path}")
                    print(f"Error: {e}")
                    commits = None

                if commits:
                    latest_commit_sha = commits[0].sha
                    try:
                        file_content = ghapi_client.repos.get_content(
                            path=truncated_file_path, ref=latest_commit_sha
                        )
                        decoded_content = base64.b64decode(file_content.content).decode(
                            "utf-8"
                        )
                        num_lines = len(decoded_content.splitlines())

                        # Check if the file is too large
                        if num_lines > 1000:
                            print(f"File {file_path} is too large, skipping...")
                            relevant_documents.remove(document)
                            continue

                        file_objects += (
                            f"Code for the following {file_path}: \n {decoded_content} "
                        )
                    except Exception as e:
                        print(f"Failed to get file content for file: {file_path}")
                        print(f"Error: {e}")
                        continue
                else:
                    print(f"No commits found for file: {file_path}")

            kwargs["file_objects"] = file_objects

            file = generate_code(
                file, card.type, template_name, variables, relevant_documents, **kwargs
            )
        else:
            template_str = load_template_from_file(
                TEMPLATE_FILEPATHS[card.type][template_name], variables
            )
            metadata[template_name] = run_llm_chain(template_str, **kwargs)

    pull_request_url = create_github_pull_request(
        ghapi_client, ghapi_raja_client, file, metadata
    )

    return pull_request_url
