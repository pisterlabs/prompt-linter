import os
from langchain.tools import tool
from datetime import datetime
from glob import glob



def clock() -> str:
    'Get the current time with this tool'
    return datetime.now().strftime("%Y-%m-%d %I:%M%p")


def safePath(path: str) -> str:
    nor_path = os.path.normpath(path)

    if nor_path.startswith('/') or nor_path.startswith('\\'):
        nor_path = os.path.join(os.path.normpath(os.environ.get('gpt_working_dir')),
                                os.path.normpath(path[1:]))

    nor_path = os.path.relpath(nor_path)
    return nor_path


@tool("clock", return_direct=False)
def utc_clock() -> str:
    'Get the current time with this tool'
    return clock()


@tool("file_reader")
def file_reader(filename: str, filepath: str = "./") -> str:
    """
    Read content from file.
    Returns content of the file or <<eof>>
    """
    try:
        file_path = safePath(os.path.join(filepath, filename))
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return "<<eof>>"


@tool("file_writer")
def file_writer(filename: str, filepath: str = "./", content: str = "") -> str:
    """
    Write content to file.
    Returns True or exception
    """
    try:
        file_path = safePath(os.path.join(filepath, filename))
        if not os.path.isdir(file_path):
            os.makedirs(safePath(filepath))
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


@tool("mkdir")
def mkdir(path: str, directory: str) -> bool:
    """
    Create directory at target path
    Returns True of success, otherwise False
    """
    try:
        file_path = safePath(os.path.join(path, directory))
        os.mkdir(file_path)
        return True
    except Exception as e:
        return "<<eof>>"


@tool("use_glob")
def use_glob(path: str = "./*") -> str:
    """
      This tool returns glob.glob(path)
    """
    file_path = safePath(path)
    return str(glob(file_path))
