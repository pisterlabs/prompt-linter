from dotenv import load_dotenv
import pygetwindow as gw
import pyautogui
import win32file
import pywintypes
import json
import openai
import os
import re
import black

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class VSCodeModule():
    def __init__(self, parser_class):
        self.parser_class = parser_class
        self.init_parser_class()

    def init_parser_class(self):
        self.parser_class.add_object(self.parser_obj)

    @property
    def parser_obj(self):
        return {
            "keywords": ["my code", "active code", "this code"],
            "prior_function": [
                {
                    "function": self.parse_fn,
                    "arg": {}
                }
            ]
        }

    def parse_fn(self, arg):
        vscode_data = self.request()
        
        keywords = {
            "refactor_keyword": ["modify", "change", "rewrite", "fix"],
            "add_keyword": ["write", "add", "extend"]
        }

        all_keyword = keywords["refactor_keyword"] + keywords["add_keyword"]

        action_keyword = ""
        for word in all_keyword:
            if word in arg["prompt"]:
                for type in keywords:
                    if word in keywords[type]:
                        action_keyword = type
                        break

        code = vscode_data["highlightedCode"] if vscode_data["highlightedCode"] else vscode_data["activeCode"]
        
        if action_keyword == "refactor_keyword":
            pass
            
        elif action_keyword == "add_keyword":
            pass

        elif action_keyword == "":
            return {
                "messages": [
                    {
                        "role": "system", "content": f"Code obtained from VS Code: {code}"
                    },
                ],

                "histories": [{
                    "role": "system",
                    "content": f"Code obtained from VS Code: {code}"
                }],
                "wrapper_functions": [self.test]
            }

    def test(self, chunk):
        print(chunk)

    def request(self):
        window_title = gw.getWindowsWithTitle(pyautogui.getActiveWindow().title)
        pipe_code_match = re.search(r'\s-\s(.*?)\s-\s', window_title[0].title)
        
        PIPE_NAME = f'\\\\.\\pipe\\{pipe_code_match.group(1)}'
        
        while True:
            try:
                handle = win32file.CreateFile(
                    PIPE_NAME,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0, None,
                    win32file.OPEN_EXISTING,
                    0, None)
                break
            except pywintypes.error as e:
                if e.args[0] == 2: # File not found error
                    continue
                raise
            
        # Connect and receive the message

        message_to_send = "getData".encode('utf-8')
        win32file.WriteFile(handle, message_to_send)

        _, data = win32file.ReadFile(handle, 100000)
        win32file.CloseHandle(handle)
        print('Received data from VSCODE')
        
        raw_data = data.decode('utf-8')
        dict_data = json.loads(raw_data)
        return dict_data


def get_vscode(content_obj):
    print("CALLED GET_VSCODE")
    data = json.loads(request("getData"))

    new_string = f"{content_obj['label']}:{data[content_obj['get_method']]}"
    keyword_used = content_obj["keyword_obj"]["keyword_used"]
    new_prompt = content_obj["prompt"].replace(keyword_used, new_string)

    result_obj = {"filePath": data["filePath"], "prompt": new_prompt}

    # Check if word trigger that will overwrite the current path file, if there is, add a write_vscode function in the data
    trigger_words = [
        "rewrite",
        "refactor",
        "overwrite",
        "change",
        "remove",
        "replace",
        "reformat",
        "write a code"
    ]

    for word in trigger_words:
        if word in content_obj["prompt"]:
            result_obj["fn_call"] = {
                "name": "request.add_message",
                "arg": {
                    "role": "system",
                    "content": "When providing a code snippet, always enclose it within the designated VSCode wrapper, formatted as !!!vscode-start!!!<content>!!!vscode-end!!!. Its essential to ensure the code inside this wrapper adheres strictly to its original format. This means any spaces, newlines, or other characters should remain unchanged, and no additional elements should be introduced unless specified. For instance, if tasked with changing the servant value to 'Abby', age to 21, and city to 'Dallas' in the code {'servant': 'Bryab', 'age': 23, 'city': 'Houston'},, your wrapped code should appear as !!!vscode-start!!!{'servant': 'Abby', 'age': 21, 'city': 'Dallas'},!!!vscode-end!!!, noting the retained comma at the end. Return the whole code, don't add in response like 'Rest of your code remains unchanged, it has to be complete' because this will replace the original code. Alert the user that you are writing the changes now in VS Code first then the code snippet with wrapper.",
                },
            }

            # Invoke additional prior function
            '''
            result_obj["post_functions"] = [
                {
                    "name": "write_vscode",
                    "arg": {
                        "item_path": data["filePath"],
                        "vscode_get_method": content_obj["get_method"],
                        "highlighted_code": {
                            "content": data["highlightedCode"],
                            "start_line": data["highlightedCodeStartLine"],
                            "end_line": data["highlightedCodeEndLine"],
                        },
                    },
                }
            ]
            '''
            
            # Append a wrapper perser object
            result_obj["wrapper_parsers"] = [
                {
                    "opening_tag": "!!!vscode-start!!!",
                    "closing_tag": "!!!vscode-end!!!",
                    "function": insert_code,
                    "arg": {
                        "item_path": data["filePath"],
                        "highlighted_code": data["highlightedCode"]
                    }
                }
            ]
            
            if content_obj["get_method"] == "activeCode":
                result_obj["wrapper_parsers"][0]["arg"]["start_line"] = 0
                remove_code(data["filePath"])
                
            elif content_obj["get_method"] == "highlightedCode":
                result_obj["wrapper_parsers"][0]["arg"]["start_line"] = remove_code(data["filePath"], data["highlightedCode"])
                
            result_obj["post_functions"] = [
                {
                    "name": format_code,
                    "arg": {
                        "item_path": data["filePath"]
                    }
                }
            ]
                
    print(f"RESULT OBJ: {result_obj}")
    return result_obj

def write_vscode(obj):
    content = obj["prompt"]
    pattern = re.compile(r"!!!vscode-start!!!(.+?)!!!vscode-end!!!", re.DOTALL)

    tag_match = re.search(pattern, content)

    if tag_match:
        code = tag_match.group(1)

        # remove the OPENAI code format
        clean_code = clean_string(code)

    print(obj)
    # Refactor the code with given item path
    if "item_path" in obj:
        item_path = obj["item_path"]

        # Get the method used in vscode get method if partial or full code was taken
        if obj["vscode_get_method"]:
            # if full code, replace the whole code with the new code
            if obj["vscode_get_method"] == "activeCode":
                with open(item_path, "w") as f:
                    f.write(clean_code)

            elif obj["vscode_get_method"] == "highlightedCode":
                pos = remove_code(item_path, obj["highlighted_code"]["content"])
                insert_code(item_path, clean_code, pos)
                format_code("python", item_path)

    # Else open a new vscode window and put new code inside


def find_string_line(filename, target_string, position):
    start_line = None
    end_line = None

    print(f"this is the target string: {target_string}")
    if position not in ["start", "end"]:
        return "Invalid position argument. Use 'start' or 'end'."

    with open(filename, "r") as f:
        for line_number, line_content in enumerate(f, 1):
            if target_string in line_content:
                if start_line is None:
                    start_line = line_number
                end_line = line_number

    if position == "start":
        return start_line
    elif position == "end":
        return end_line


def write_to_line(filename, line_number, string_to_write):
    lines = []
    print("WRITING TO LINE")
    with open(filename, "r") as f:
        lines = f.readlines()

    # Update the specific line
    lines[line_number - 1] = string_to_write + "\n"

    # Write back to file
    with open(filename, "w") as f:
        f.writelines(lines)


def format_code(obj):
    filename = obj["item_path"]
    with open(filename, "r") as f:
        code = f.read()

    formatted_code = black.format_str(code, mode=black.FileMode())

    with open(filename, "w") as f:
        f.write(formatted_code)


def clean_string(s):
    # Trim spaces and newlines from the start and end
    s = s.strip()

    # Remove code format tags from the beginning and end
    s = re.sub(r"^```[a-zA-Z]*\n|```$", "", s)

    return s


def remove_code(filename, code = ""):
    print(f" this is the filename: {filename}, and this is the code: {code}")
    
    # Normalize newlines function
    def normalize_newlines(string):
        return string.replace('\r\n', '\n')

    if code:
        with open(filename, "r") as file:
            file_contents = file.read()

        # Normalize the newlines for both file contents and code
        file_contents = normalize_newlines(file_contents)
        code = normalize_newlines(code)

        # Find the position of the provided code
        position = file_contents.find(code)

        # If the code is found, remove it
        if position != -1:
            file_contents = file_contents.replace(
                code, "", 1
            )  # Only replace the first occurrence

            # Write the modified contents back to the file
            with open(filename, "w") as file:
                file.write(file_contents)

        return position
    
    else:
        with open(filename, "w") as f:
            f.write("")
            
            return 0



def insert_code(obj):
    filename = obj["item_path"]
    code = obj["chunk"]
    start_position = obj["start_line"]
    
    with open(filename, "r") as file:
        file_contents = file.read()

    # Insert the code at the specified position
    before = file_contents[:start_position]
    after = file_contents[start_position:]
    modified_contents = before + code + after

    # Write the modified contents back to the file
    with open(filename, "w") as file:
        file.write(modified_contents)