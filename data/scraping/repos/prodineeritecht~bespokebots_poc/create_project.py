from __future__ import annotations
from typing import Any, List, Optional, Type
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from bespokebots.services.agent.todoist_tools.base import TodoistBaseTool
from bespokebots.models.tools.todoist import (
    TodoistProject, 
    TodoistTask,
    TodoistDue
)


class CreateProjectSchema(BaseModel):

    name: str = Field(
        ...,
        title="Project Name",
        description="The name of the project."
    )

    parent_id: Optional[int] = Field(
        None,
        title="Parent ID",
        description="The ID of the parent project. If not specified, the project will be top-level."
    )

    color: Optional[str] = Field(
        None,
        title="Color",
        description="The color of the project. If not specified, the default color will be used."
    )

    is_favorite: Optional[bool] = Field(
        None,
        title="Is Favorite",
        description="Whether the project should be marked as favorite. If not specified, the project will not be marked as favorite."
    )

    view_style: Optional[str] = Field(
        None,
        title="View Style",
        description="Whether the project should be displayed as a list or a board in each of the Todoist clients. List style is the default view style."
    )

class CreateProjectTool(TodoistBaseTool):
        
        name: str = "create_project"
        description: str = """Use this tool to create a project in Todoist."""

        args_schema: Type[CreateProjectSchema] = CreateProjectSchema

        def _run(
            self,   
            name: str,
            parent_id: Optional[int] = None,
            color: Optional[str] = None,
            is_favorite: Optional[bool] = None,
            view_style: Optional[str] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> dict:
            """Create a project in Todoist."""
            try:
                 # Grab the current local variables
                kwargs = locals()
                # Remove self and run_manager from kwargs
                kwargs.pop('self')
                kwargs.pop('run_manager')
                # Now remove any variables that are None
                kwargs = {k: v for k, v in kwargs.items() if v is not None}
                project = self.todoist_client.add_project(**kwargs)
                return TodoistProject.from_todoist(project).dict()
            except Exception as e:
                raise Exception(f"An error occurred when trying to create a project: {e}")
                

        async def _arun(self, *args: Any, **kwargs: Any) -> dict:
                raise NotImplementedError(f"The tool {self.name} does not support async yet.")