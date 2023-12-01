import json
import weaviate
import os
import openai

from github import Github
from github import Auth
from weaviate.util import generate_uuid5

class RicaiCodestoreHelper:
    def __init__(self, weaviate_url, weaviate_key, openai_key, github_token, github_user):
        """
        Initializes the RicaiCodestoreHelper with the provided Weaviate and Github credentials.
        Args:
            Weaviate_url (str): weaviate database connection url.
            Weaviate_api_key (str): weaviate database connection API key.
            Openai_api_key (str): OpenAI API key.
        """
        self.w_client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=weaviate.AuthApiKey(
                api_key=weaviate_key
            ),
            additional_headers={
                "X-OpenAI-Api-Key": openai_key
            },
        )

        self.openai_key = openai_key

        class_obj = {
            "class": "Codefile",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "davinci",
                    "modelVersion": "003",
                    "type": "text"
                }
            },
            "properties": [
                {
                    "name": "file_path",
                    "dataType": ["text"],
                    "description": "Path to code file",
                },
                {
                    "name": "github_url",
                    "dataType": ["text"],
                    "description": "Github url of code file",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True
                        }
                    },
                },
                {
                    "name": "type",
                    "dataType": ["text"],
                    "description": "Type of the file",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True
                        }
                    },
                },
                {
                    "name": "repo",
                    "dataType": ["text"],
                    "description": "The code repository in Github",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True
                        }
                    },
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "File content (code)",
                },
            ],
        }

        schemas: list = self.w_client.schema.get()["classes"]
        if len(schemas) != 0:
            schema = schemas[0]["class"]  
            if schema != "Codefile":
                self.w_client.schema.create_class(class_obj)

        # Github setup
        ghub_auth = Auth.Token(github_token)
        ghub = Github(auth=ghub_auth)
        self.ghub_user = ghub.get_user(github_user)

    def upsert_codebase(self, ghub_repo_name):
        repo = self.ghub_user.get_repo(ghub_repo_name)
        contents = repo.get_contents("")
        codefiles = []

        # TODO: populate with more ignorable file types
        ignore_extensions = [
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".ico", ".svg",
            ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv",
            ".mp3", ".wav", ".ogg", ".aac", ".flac", ".wma",
            ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".ods", ".odp",
            ".zip", ".rar", ".tar", ".gz", ".7z",
            ".exe", ".dll", ".app", ".apk", ".iso", ".img", ".dmg",
        ]

        while contents:
            file = contents.pop(0)
            if file.type == "dir":
                contents.extend(repo.get_contents(file.path))
            else:
                extension = os.path.splitext(file.name)[1].lower()
                if extension not in ignore_extensions:
                    print(file.path)
                    file_content = file.decoded_content.decode("utf-8")
                    codefiles.append({
                        "file_path": file.path,
                        "github_url": file.url,
                        "type": file.type,
                        "repo": file.repository.name,
                        "content": file_content
                    })

        # TODO: check if code from specific repo/codebase in Github is already present in the vector database
        # TODO: make sure that deterministic uuid generation works 
        class_name = "Codefile"
        with self.w_client.batch() as batch:
            for codefile in codefiles:
                batch.add_data_object(
                    codefile, 
                    class_name,
                    uuid=generate_uuid5(identifier=codefile["file_path"], namespace=codefile["repo"])
                )
        return True

    def retrieve_all_code(self, ghub_repo_name):
        response = (
            self.w_client.query
                .get("Codefile", ["file_path", "github_url", "type", "repo", "content"])
                .with_where({
                    "path": ["repo"],
                    "operator": "Like",
                    "valueText": f"*{ghub_repo_name}*"
                })
                .do()
        )
        files = response["data"]["Get"]["Codefile"]
        return json.dumps(files)

    def retrieve_latest_commit_code(self, ghub_repo_name, sha):
        repo = self.ghub_user.get_repo(ghub_repo_name)
        commit = repo.get_commit(sha)
        # print(commit.commit.message)
        # print(commit.commit.author.email)
        relevant_files = []
        for file in commit.files:
            if file.status != "removed":
                relevant_files.append(file.filename)

        response = (
            self.w_client.query
                .get("Codefile", ["file_path", "github_url", "type", "repo", "content"])
                .with_where({
                    "path": ["repo"],
                    "operator": "Like",
                    "valueText": f"*{ghub_repo_name}*"
                })
                .do()
        )
        files = response["data"]["Get"]["Codefile"]
        result = []
        for f in files:
            if f["file_path"] in relevant_files:
                result.append(f)
        return json.dumps(result)

    def semantic_code_search(self, ghub_repo_name, context):
        openai.api_key = self.openai_key
        vectorized_context = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=context
        )["data"][0]['embedding']

        response = (
            self.w_client.query
                .get("Codefile", ["file_path", "github_url", "type", "repo", "content"])
                .with_near_vector(vectorized_context)
                .with_where({
                    "path": ["repo"],
                    "operator": "Like",
                    "valueText": f"*{ghub_repo_name}*"
                })
                .do()
        )
        files = response["data"]["Get"]["Codefile"]
        return json.dumps(files)

    def retrieve_code_by_location(self, ghub_repo_name, location):
        where_filter = {
            "operator": "And",
            "operands": [
                {
                    "path": ["repo"],
                    "operator": "Like",
                    "valueText": f"*{ghub_repo_name}*"
                },
                {
                    "path": ["file_path"],
                    "operator": "Like",
                    "valueText": f"*{location}*"
                }
            ]
        }
        response = (
            self.w_client.query
                .get("Codefile", ["file_path", "github_url", "type", "repo", "content"])
                .with_where(where_filter)
                .do()
        )
        files = response["data"]["Get"]["Codefile"]
        return json.dumps(files)