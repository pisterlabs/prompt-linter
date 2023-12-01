import os
from pprint import pprint

from git import Repo
from langchain.document_loaders import GitLoader

from commons import DEBUG, std_response
from vector_index import VectorEntry


def remove_dir(local_temp_path):
    if local_temp_path == '/':
        raise Exception("Cannot use / as a temp path")
    if not os.path.exists(local_temp_path):
        return
    if os.path.isfile(local_temp_path):
        os.remove(local_temp_path)
        return
    for root, dirs, files in os.walk(local_temp_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def get_repo(repo_url, branch):
    response = std_response()
    repo_name = repo_url.rsplit('/', 1)[1]
    if repo_url[:8] == "https://":
        repo_type = "remote"
        # https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/git
        local_temp_repo_path = f"/tmp/{repo_name}"
        try:
            remove_dir(local_temp_repo_path)
        except Exception as err:
            if DEBUG:
                raise
            response["error"] = True
            response["error_msg"] = f"ERROR: {str(err)}"
            return response
        try:
            repo = Repo.clone_from(
                url=repo_url,
                to_path=local_temp_repo_path
            )
        except Exception as err:
            if DEBUG:
                raise
            response["error"] = True
            response["error_msg"] = f"ERROR: {str(err)}"
            return response
    else:
        repo_type = "local"
        try:
            local_temp_repo_path = repo_url
            repo = Repo(local_temp_repo_path)
        except Exception as err:
            if DEBUG:
                raise
            response["error"] = True
            response["error_msg"] = f"ERROR: {str(err)}"
            return response
    if not branch:
        try:
            branch = repo.head.reference
        except Exception as err:
            if DEBUG:
                raise
            response["error"] = True
            response["error_msg"] = f"ERROR: {str(err)}"
            return response
    print()
    print(f"Github repository URL: {repo_url}")
    print(f"Branch (default 'main'): {branch}")
    try:
        loader = GitLoader(
            repo_path=f"{local_temp_repo_path}/",
            branch=branch,
        )
    except Exception as err:
        if DEBUG:
            raise
        response["error"] = True
        response["error_msg"] = f"ERROR: {str(err)}"
        return response

    response["data"] = loader.load()

    # Add some context for the repo
    metadata = {
        "source": repo_url,
        "branch": branch,
        "repo_type": repo_type,
        "repo_name": repo_name,
        "repo_path": local_temp_repo_path,
        "comments": ",".join([
                    "this is the context of the:",
                    "repo context",
                    "repository context",
                    "supplied repo"
                    "supplied repository",
                    "loaded repo",
                    "loaded repository",
                    "supplied git repo",
                    "supplied git repository",
                    "loaded git repo",
                    "loaded git repository",
                    "supplied branch",
                    "loaded branch",
            ])
    }
    response["data"].append(
        VectorEntry(
            page_content=f"The supplied git repo name is {repo_name}",
            metadata=metadata
        )
    )
    response["data"].append(
        VectorEntry(
            page_content=f"{repo_type} git repo content of {repo_url}",
            metadata=metadata
        )
    )

    if DEBUG:
        print(f"Repo: {repo_url}")
        print(f"Branch: {branch}")
        print("Data:")
        pprint(response["data"])
    return response


def get_repo_data(repo_url, branch):
    if repo_url == "":
        return []
    repo_response = get_repo(repo_url, branch)
    if repo_response["error"]:
        return []
    repo_data = repo_response["data"]
    return repo_data
