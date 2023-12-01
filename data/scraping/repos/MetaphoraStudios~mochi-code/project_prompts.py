"""Utilities to create project specific prompts."""
import pathlib
from typing import Optional

from langchain import PromptTemplate

from mochi_code.code.mochi_config import (get_project_details_path,
                                          load_project_details,
                                          search_mochi_config)

_ProjectTemplate = PromptTemplate(
    input_variables=["language", "package_manager", "dependencies"],
    template=
    "The user is working on a {language} project using {package_manager} as " +
    "a package manager. Here's a list of dependencies used by the project: " +
    "({dependencies})",
)


def get_project_prompt(start_path: pathlib.Path) -> Optional[str]:
    """Get the project prompt if available.

    Args:
        start_path (pathlib.Path): The path to search for the project details
        (uses search_mochi_config to find the project details).

    Returns:
        str: The project prompt or None if there are no project details
        available.
    """
    existing_root = search_mochi_config(start_path)
    if not existing_root:
        return None

    project_details_path = get_project_details_path(existing_root)
    project_details = load_project_details(project_details_path)

    return _ProjectTemplate.format(
        language=project_details.language,
        package_manager=project_details.package_manager,
        dependencies=project_details.dependencies)
