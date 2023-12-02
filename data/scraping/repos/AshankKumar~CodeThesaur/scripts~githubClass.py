from openai_function_call import OpenAISchema
from pydantic import Field

class githubProject(OpenAISchema):
    "Correctly extracted information from the html of a github project"
    name: str = Field(..., description="The name of the project. Don't include the author's name or github in this field")
    url: str = Field(..., description="The github url of the project")
    author: str = Field(..., description="The author of the project")
    description: str = Field(..., description="A short summary describing the project")

class projectList(OpenAISchema):
    "List of correctly extracted github projects"
    projects: list[githubProject]


