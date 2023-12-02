from folder_config import load_config, FolderConfig
from load_file import FileInfo
from langchain.llms import OpenAI
import dotenv
import os
import shutil

def get_prompt(config:FolderConfig, file:FileInfo):
    return f"""The following contains the configuration of this folder:
{config.description()}
The following contains the information about the file:
{file.describe()}
Please choose a directory to move the file to:
"""

def get_category(config:FolderConfig, file:FileInfo):
    dotenv.load_dotenv()
    llm = OpenAI()
    prompt = get_prompt(config, file)
    res = llm(prompt)
    # TODO: improve the validation part
    for d in config.dirs:
        if d.path in res:
            return d.path
    return None

if __name__ == "__main__":
    config = load_config(input("Please enter the path to the config file: "))
    file = FileInfo(input("Please enter the path to the file: "))
    category = get_category(config, file)
    if category is not None:
        shutil.move(file.filename, category)
