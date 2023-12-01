from openai import OpenAI
from credentials import openai_key
import os


def initiate_client():
    """
    Initializes and returns an OpenAI client using the provided API key.

    Returns:
    OpenAI: An instance of the OpenAI client configured with the specified API key.
    """
    client = OpenAI(api_key=openai_key)
    return client


def get_all_files_in_path(file_path):
    """
    Returns a list of all file paths within the specified directory and its subdirectories.
    Skips '.DS_Store' files common on macOS.

    Args:
    - directory (str): The path to the directory.

    Returns:
    - list: A list of file paths.
    """
    all_file_paths = []
    for root, _, files in os.walk(file_path):
        for file in files:
            if file != '.DS_Store':
                all_file_paths.append(os.path.join(root, file))
    return all_file_paths


# Additional function to select a file for upload
def select_file_for_upload(file_path):
    """
    Presents a list of all files in the specified path and allows the user to select a file for upload.

    Parameters:
    file_path (str): The path where files are located.

    Returns:
    str: The path of the selected file, or None if the selection is invalid.
    """
    all_files = get_all_files_in_path(file_path)
    print("Please select a file to upload:")
    for idx, file_name in enumerate(all_files):
        print(f"{idx + 1}. {file_name}")
    selected_index = int(input("Enter the number of the file you want to upload: ")) - 1
    if 0 <= selected_index < len(all_files):
        return all_files[selected_index]
    else:
        print("Invalid selection.")
        return None


# Sample assistant template used for creating new assistants in the system.
new_assistant = {
    "model": "gpt-4-1106-preview",
    "name": "Laura",
    "instructions": """You are the ultimate librarian, you know everything about the files attached to you
     and you review them and answer primarily from their content""",
    "tools": [{"type": "code_interpreter"}, {"type": "retrieval"}],
    "description": "The ultimate librarian",
    "file_ids": []
}
