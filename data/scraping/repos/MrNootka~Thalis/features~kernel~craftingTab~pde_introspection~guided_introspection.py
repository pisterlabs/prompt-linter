from typing import List
from pathlib import Path

import os
import openai
from dotenv import load_dotenv

load_dotenv()

def create_folder_structure_file(start_path: str, output_file: str) -> None:
    with open(output_file, "w") as f:
        for root, _, files in os.walk(start_path):
            if "__pycache__" not in root and ".git" not in root and ".misc" not in root:
                for file in files:
                    path = os.path.join(root, file)
                    f.write(f"{path}\n")

def get_all_lines_from_file(file_path: Path) -> List[str]:
    return file_path.read_text().splitlines()

def ask_user_option() -> str:
    while True:
        user_input = input("Do you want to consider a whole folder (type 'yes'), list the specific files (type 'no') or specify specific folder (type directory path)? ")
        if user_input.lower() == "yes" or user_input.lower() == "no" or os.path.isdir(user_input):
            return user_input
        else:
            print("Invalid input, please enter 'yes', 'no' or a valid directory path.")

def get_files_to_consider() -> List[str]:
    base_path = Path(__file__).absolute().parent
    all_files = get_all_lines_from_file(base_path / "1_folder_structure.txt")
    print("Files in the folder structure:")
    [print(f"{i}. {_file}") for i, _file in enumerate(all_files)]
    return get_valid_user_input_for_files(all_files)

def get_valid_user_input_for_files(all_files: List[str]) -> List[str]:
    while True:
        user_input = input("\nEnter file numbers to consider (comma-separated): ")
        if is_valid_user_input(user_input, len(all_files)):
            break
        else:
            print("Invalid input. Please enter valid file numbers separated by commas.")
    return [all_files[int(i)] for i in user_input.split(",")]

def is_valid_user_input(user_input: str, max_limit: int) -> bool:
    return all(x.isdigit() for x in user_input.split(",")) and all(0 <= int(x) < max_limit for x in user_input.split(","))

def save_code_files(files: List[str], output_file: str) -> None:
    with output_file.open("w") as f:
        for _file in files:
            try:
                content = Path(_file).read_text().strip()
                if content:
                    f.write(f"### {_file}\n```\n{content}\n```\n")
                else:
                    f.write(f"### {_file} (Empty)\n")
            except FileNotFoundError as e:
                print(f"Error reading file '{_file}': {e}")

def save_prompt(user_prompt: str, output_file: Path) -> None:
    output_file.write_text(f"## User Prompt\n{user_prompt}")

def merge_files(input_files: List[str], output_file: Path) -> None:
  
    base_path = Path(__file__).absolute().parent
    with open(output_file, "w") as out_file:
        out_file.write("\n-----\n\n".join([base_path.joinpath(_file).read_text().strip() 
                                          if base_path.joinpath(_file).exists() 
                                          else f"{_file} (Empty)"
                                          for _file in input_files]))

def get_user_prompt(filepath: Path) -> str:
    if filepath.exists():
        last_prompt = filepath.read_text().strip()
        if last_prompt:
            user_input = input(f"\nLast used prompt: {last_prompt}\nDo you want to reuse the last prompt? (Type 'reuse' or enter a new prompt): ").strip()
            if user_input.lower() == "reuse":
                return last_prompt
    return input("\nPlease enter the user prompt: ").strip()

def get_api_response(user_data: str) -> str:  
    openai.api_key = os.getenv("OPENAI_API_KEY")


    response = openai.ChatCompletion.create(
       model="gpt-4",
       messages=[
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": user_data}
       ]
    )
    return response.choices[0].message['content']

def guided_introspection():
    base_path = Path(__file__).absolute().parent

    # Section to ask user
    user_choice = ask_user_option()

    if user_choice.lower() == "yes":
        start_folder = "."
        is_whole_folder = True
    elif user_choice.lower() == "no":
        start_folder = "."
        is_whole_folder = False
    else:
        start_folder = user_choice
        is_whole_folder = True
        
    create_folder_structure_file(start_folder, base_path / "1_folder_structure.txt")
    
    # Check user's choice
    if is_whole_folder:
        # consider all files
        all_files = get_all_lines_from_file(base_path / "1_folder_structure.txt")
        files_to_consider = [file for file in all_files if "__pycache__" not in file and ".git" not in file and ".misc" not in file]
        print(f"All files ({len(files_to_consider)} total) are considered.")
    else:
        # consider selected files
        files_to_consider = get_files_to_consider()

    save_code_files(files_to_consider, base_path / "2_code.txt")

    prompt_file = base_path / "3_prompt.txt"
    user_prompt = get_user_prompt(prompt_file)
    save_prompt(user_prompt, prompt_file)

    output_file = base_path / "files_and_prompt.txt"
    merge_files(["1_folder_structure.txt", "2_code.txt", "3_prompt.txt"], output_file)
    
    api_response = get_api_response(output_file.read_text().strip())
    with open(base_path / "gpt_response.txt", "w") as file:  
        file.write(api_response)

def clean_files():
    base_path = Path(__file__).absolute().parent
    files_to_clean = ["1_folder_structure.txt", "2_code.txt", "3_prompt.txt", "gpt_response.txt", "files_and_prompt.txt"]

    for filename in files_to_clean:
        filepath = base_path / filename
        if filepath.exists():
            os.remove(filepath)

if __name__ == "__main__":
    clean_files()
    guided_introspection()
