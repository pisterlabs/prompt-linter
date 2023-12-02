
import os
import sys
from typing import List

from langchain.schema import Document
from integrations.source_control_base import SourceControlBase, CodeComment

# Append the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


class FileIntegration(SourceControlBase):
    def __init__(self):
        # Get the output file from the environment variable or use the default
        self.output = os.environ.get("FILE_OUTPUT", "output.md")

    def commit_changes(self, source_branch, target_branch, commit_message, code_documents: List[dict]):
        # Iterate over the metadata of each document
        for metadata in code_documents['metadatas']:
            # Create a path for the refactored file
            path = metadata['file_path'] + ".refactored.py"
            
            # Open the file in append mode if it exists, otherwise create a new file
            mode = 'a' if os.path.exists(path) else 'w'
            with open(path, mode) as file:
                file.write(metadata['refactored_code'])

    def add_pr_comments(self, comments: List[CodeComment]):
        last_file = ''
        output_string = ''
        for comment in comments:
            # Add a new section for each file
            if comment.file_path != last_file:
                output_string += f"\n\n## {comment.file_path}\n"
                last_file = comment.file_path

            # Add the line numbers if they exist
            if comment.start is not None:
                output_string +=  f"- **Lines: {comment.start}-{comment.end}**: "
            
            # Add the comment
            output_string +=  comment.comment + "\n"

        # Write the comments to the output file
        with open(self.output, "w") as f:
            f.write(output_string)

