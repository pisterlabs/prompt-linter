""" Util that calls Github. """
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from prompts import (
    GITHUB_READ_REPOS,
    GITHUB_CREATE_BRANCH,
)
from langchain.utils import get_from_dict_or_env


class GithubApiWrapper(BaseModel):
    """ Wrapper for Github API. """

    github: Any  #: :meta private:

    # List of operations that this tool can perform
    operations: List[Dict] = [
        {
            "mode": "repo_read",
            "name": "Read the repos",
            "description": GITHUB_READ_REPOS,
        },
        {
            "mode": "branch_create",
            "name": "Create a branch",
            "description": GITHUB_CREATE_BRANCH,
        }
    ]

    class Config:
        """ Configuration for this pydantic object. """
        extra = Extra.forbid

    def list(self) -> List[Dict]:
        return self.operations

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """ Validate that api key and python package exists in environment. """
        # token = get_from_dict_or_env(values, "token", "GITHUB_TOKEN")
        # values["github"] = github

        return values
    
    def run(self, mode: str, text: Optional[str]) -> str:
        """ Based on the mode from the caller, run the appropriate function. """
        if mode == "repo_read":
            return self.repos_read()
        elif mode == "branch_create":
            return self.branch_create(text)
        else:
            raise ValueError(f"Got unexpected mode {mode}")

    def repos_read(self) -> str:
        """ Read the rpos from the github workspace """
        try:
            import json
            # Call the github to get repos

        except ImportError:
            raise ImportError(
                "json is not installed. " "Please install it with `pip install json`"
            )
        except Exception as e:
            print("Error: {}".format(e))
            raise Exception("Failed to read repos")

    def branch_create(self, query: str) -> str:
        """ Create a branch in the github workspace """
        try:
            import json
            # Call the github to create a branch

        except ImportError:
            raise ImportError(
                "json is not installed. " "Please install it with `pip install json`"
            )
        except Exception as e:
            print("Error: {}".format(e))
            raise Exception("Failed to create branch")
        
