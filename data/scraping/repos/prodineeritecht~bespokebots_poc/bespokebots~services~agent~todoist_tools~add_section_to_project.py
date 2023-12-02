from __future__ import annotations
from typing import Any, List, Optional, Type
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from bespokebots.services.agent.todoist_tools.base import TodoistBaseTool
from bespokebots.models.tools.todoist import TodoistProject, TodoistTask, TodoistSection


class AddSectionToProjectSchema(BaseModel):
    project_id: str = Field(
        ...,
        title="Project ID",
        description="The ID of the project to add the section to.",
    )

    section_name: str = Field(
        ..., title="Section Name", description="The name of the section to add."
    )

    order: Optional[int] = Field(
        None,
        title="Order",
        description="The order of the section inside the project. The smallest value is 1.",
    )


class AddSectionToProjectTool(TodoistBaseTool):
    name: str = "add_section_to_project"
    description: str = """Use this tool to add a section to a project in Todoist."""

    args_schema: Type[AddSectionToProjectSchema] = AddSectionToProjectSchema

    def _run(
        self,
        project_id: str,
        section_name: str,
        order: Optional[int] = None,
        **kwargs: Any,
    ) -> dict:
        section = (
            self.todoist_client.add_section(
                project_id=project_id, name=section_name, order=order
            )
            if order
            else self.todoist_client.add_section(
                project_id=project_id, name=section_name
            )
        )

        return TodoistSection.from_todoist(section).dict()
    
    async def _arun(self, *args: Any, **kwargs: Any) -> dict:
                raise NotImplementedError(f"The tool {self.name} does not support async yet.")
