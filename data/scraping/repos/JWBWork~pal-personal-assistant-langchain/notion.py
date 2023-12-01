import os
import re
from pprint import pformat
from typing import Any, Dict, List, Optional

# from pydantic.v1 import BaseModel
from langchain.tools import StructuredTool, tool
from loguru import logger
from notion_client.errors import APIResponseError
from pydantic.v1 import BaseModel, Field, root_validator

# from src.notion_v1 import DATABASE_ID, NotionTodoCli, NotionTodo, NotionPropertyConstructors
from src.notion_v1 import NotionPropertyConstructors, NotionTodo, NotionTodoCli

DATABASE_ID = os.environ['DATABASE_ID']
_notion = NotionTodoCli(
    database_id=DATABASE_ID,
    notion_token=os.environ['NOTION_TOKEN']
)


def todo_id_is_valid(id: str) -> bool:
    return id in [t.id for t in _notion.todos]


def todo_ids_are_valid(ids: List[str]) -> bool:
    return all([id in [t.id for t in _notion.todos] for id in ids])


class TodoInput(BaseModel):
    name: str = Field(description="Todo name")
    tags: List[str] = Field(description="Tags to categorize todos")


@tool(return_direct=False, args_schema=TodoInput)
def save_todo(name: str, tags: List[str]) -> str:
    """
    Saves a todo for the user

    Infer an appropriate title. Tags can be empty if none are specified by the user.
    """
    logger.info(f"Saving todo {name} with tags {tags}")
    try:
        notion_todo = NotionTodo.new(
            notion_client=_notion,
            database_id=DATABASE_ID,
            properties={
                "Tags": tags,
                "Name": name
            }
        )
        _notion.todos.append(notion_todo)
        return f"Succesfully saved todo {notion_todo}"
    except APIResponseError as e:
        return str(e)


class SubTodoInput(TodoInput):
    parent_id: str = Field(
        description="ID for parent todo, only needed for sub-todos",
        # regex=r"[\d\w\-{0,1}]+"
    )

    # @root_validator
    # def validate_query(cls, values: Dict[str, Any]) -> Dict:
    #     parent_id = values["parent_id"]
    #     if re.match(r"[\d\w]{8}-[\d\w]{4}-[\d\w]{4}-[\d\w]{4}-[\d\w]{12}", parent_id) is None:
    #         raise ValueError(f'Invalid parent ID "{values["parent_id"]}"')
    #     return values


@tool(return_direct=False, args_schema=SubTodoInput)
def save_sub_todo(name: str, tags: List[str], parent_id: str) -> str:
    """
    Saves a child todo with a parent todo for the user

    use get_all_todos to find the best ID if a real one is unavailable.
    """
    logger.info(
        f"Saving sub-todo {name} with tags {tags} and parent_id {parent_id}")
    if not todo_id_is_valid(parent_id):
        return f"Invalid parent_id {parent_id}"

    try:
        notion_todo = NotionTodo.new(
            notion_client=_notion,
            database_id=DATABASE_ID,
            properties={
                "Tags": tags,
                "Name": name,
                "Parent todo": parent_id
            }
        )
        _notion.todos.append(notion_todo)
        return f"todo saved: {notion_todo}"
    except APIResponseError as e:
        return str(e)


@tool
def get_all_todos():
    """
    Returns a json list of all todos.

    Useful for finding an ID for an existing todo when you have to add a child todo.
    """
    logger.info("Getting all todos")
    return [dict(t) for t in _notion.todos]


@tool
def archive_todo(todo_id: str):
    """
    Archives/deletes a Todo

    Use get_all_todos to find the best ID if needed
    """
    logger.info(f"Archiving todo {todo_id}")
    if not todo_id_is_valid(todo_id):
        return f"Invalid todo_id {todo_id}"

    try:
        _notion.pages.update(
            page_id=todo_id,
            archived=True
        )
        _notion.refresh_todos()
    except APIResponseError as e:
        return str(e)


@tool
def complete_todo(todo_id: str):
    """
    Marks a Todo as complete

    Use get_all_todos to find the best ID if needed
    """
    logger.info(f"Marking todo {todo_id} as complete")
    if not todo_id_is_valid(todo_id):
        return f"Invalid todo_id {todo_id}"

    try:
        _notion.pages.update(
            page_id=todo_id,
            properties=NotionPropertyConstructors.checkbox(
                "Complete", True
            )
        )
        _notion.refresh_todos()
        return f"todo {todo_id} marked as complete"
    except APIResponseError as e:
        return str(e)


class TodoIDsInput(BaseModel):
    todo_ids: List[str] = Field(description="List of Todo IDs")


@tool(args_schema=TodoIDsInput)
def complete_todos(todo_ids: List[str]):
    """
    Useful for marking multiple Todos as complete

    Use get_all_todos to find the best IDs if needed
    """
    logger.info(f"Marking todos {pformat(todo_ids)} as complete")
    if not todo_ids_are_valid(todo_ids):
        return f"One or more invalid todo_ids {todo_ids}"

    try:
        for todo_id in todo_ids:
            _notion.pages.update(
                page_id=todo_id,
                properties=NotionPropertyConstructors.checkbox(
                    "Complete", True
                )
            )
        _notion.refresh_todos()
    except APIResponseError as e:
        return str(e)
