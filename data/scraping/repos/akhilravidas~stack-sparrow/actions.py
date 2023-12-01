"""
Supported actions
"""
from __future__ import annotations

from typing import List

from openai.types.beta import assistant_create_params
from pydantic import BaseModel

# pylint: disable=line-too-long

mark_file_reviewed: assistant_create_params.ToolAssistantToolsFunctionFunction = {
    "description": "Mark a file within the patch set as reviewed with multiple comments.",
    "name": "mark_file_reviewed",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the file to be updated.",
            },
            "accepted": {
                "type": "boolean",
                "description": "True if changes to the file looks good, False otherwise.",
            },
            "review_comments": {
                "type": "array",
                "description": "A list of comments detailing the review of the file.",
                "items": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "Description of the problem this suggestion is solving.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "The line number where the old code block starts.",
                        },
                        "old_code_block": {
                            "type": "string",
                            "description": "The old block of code that needs to be replaced.",
                        },
                        "new_code_block": {
                            "type": "string",
                            "description": "The new block of code to replace the old one.",
                        },
                    },
                    "required": [
                        "explanation",
                        "start_line",
                        "old_code_block",
                        "new_code_block",
                    ],
                },
            },
        },
        "required": ["file_path", "accepted"],
    },
}


class FileReviewComments(BaseModel):
    """
    Pydantic version of `review_comments` parameters
    """

    explanation: str
    start_line: int
    old_code_block: str
    new_code_block: str


class FileReviewResult(BaseModel):
    """
    Pydantic version of `mark_file_reviewed` parameters
    """

    file_path: str
    accepted: bool
    review_comments: List[FileReviewComments] = []  # this is fine in pydantic

    @classmethod
    def new(cls, json_input: str) -> FileReviewResult:
        """Create a new instance from JSON input"""
        return FileReviewResult.model_validate_json(json_input)


review_tool: assistant_create_params.ToolAssistantToolsFunction = {
    "function": mark_file_reviewed,
    "type": "function",
}
