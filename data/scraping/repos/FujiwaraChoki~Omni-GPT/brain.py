import os
import json
import openai

from halo import Halo
from utils.status import *
from termcolor import colored
from dotenv import load_dotenv

load_dotenv(".env")


def start():
    print(colored("I want OmniGPT to create this website, f(file): ", "magenta"), end="")
    goal = input()

    if goal.startswith("f("):
        goal = open(goal[2:-1], "r").read()

    print(colored(f"\t=> Goal: {goal}", "yellow"))

    generate(goal)


def generate(goal):
    spinner = Halo(text="Creating your website\n", spinner='dots')
    spinner.start()

    # Usually
    '''
    openai.api_key = os.getenv("OPENAI_API_KEY")
    '''

    # New
    openai.api_base = "https://api.nova-oss.com/v1"
    openai.api_key = os.getenv("NOVAAI_API_KEY")

    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"""Hello ChatGPT! I have a specific request: I want to create a website, and I need your assistance to make it happen. However, I have some clear requirements, so please stick to them precisely. Here are the details of my project:

1. **Website Purpose**: The goal of this website is to '{goal}'.

2. **File Names and Contents**: I need you to provide me with a JSON-Array containing the names of files and their respective contents. Here's an example of how it should look:
    ```json
    [
       {{
           "file_name": "",
           "file_contents": ""
       }},
       // ... (additional files)
    ]
    ```

3. **Website Type**: The website can be either a static HTML site or built using a specific framework/library like Next.js or React. Let me explain further:

   - *Static HTML Site*: This means a traditional website built using HTML, CSS, and JavaScript. Each page is a separate HTML file, and you can organize your files and directories as needed.

   - *Framework/Library like Next.js or React*: These are more modern web development tools. They allow you to build web applications with components and a structured file hierarchy. If you choose this option, you can create directories to organize your components and pages. For example, you might have a 'components' directory and a 'pages' directory to structure your project.

4. **Directory Names**: When specifying file names, include directory names in front of each filename. If a file doesn't belong to any directory, simply provide the filename. This helps in organizing your project and avoids naming conflicts.

5. **File Contents**: When it comes to file contents, it's crucial that you don't use templates. Provide the actual code. For instance, instead of commenting like '// Code to fetch openai,' you should use the real code like 'fetch(url).then(...)'. This ensures that the code is functional and ready for use.

6. **Website Requirements**: The website must include the following elements:
    - Navbar
    - Working Links
    - Data in the form of actual text (no Lorem Ipsum, but real text)
    - Footer

    If I haven't provided specific data, please fill these elements with placeholder data. This means you should create these elements with real code, not just placeholders.
7. **Coding Best Practices**: Always follow best coding practices. Ensure that there are no unterminated strings, and use double quotes consistently. Be creative and generate multiple files to demonstrate your coding skills and organization.
8. **JSON Errors**: Be vigilant to avoid any JSON-related errors. If you need to quote something within the file contents, use single quotes. This ensures that your JSON-Array is valid and error-free.
9. **Specific Output**: I only need you to provide the JSON-Array containing the file names and contents. Please refrain from including anything else in your response. This will help keep the response clean and focused on your request.
That's the detailed information for my request. Please adhere to these instructions carefully as I'm looking for a precise outcome. Thank you!""",
        temperature=0.5
    )

    create_files_and_add_contents(
        parse_response(completion["choices"][0]))

    spinner.stop()
    success(
        "Successfully created your website!")

    info("Would you like to run a webserver to see your new website? (y/n)")
    run_webserver(input())


def run_webserver(choice):
    if choice.lower() == "y":
        # Run a webserver in output directory
        spinner = Halo(
            text="Started HTTP-Server on Port 8080: http://localhost:8080")
        spinner.start()
        os.chdir("output")
        os.system("python3 -m http.server 8080 > /dev/null 2>&1")
        spinner.stop()
        info("Stopped webserver!")
    else:
        info("Aborting...")
        exit(0)


def parse_response(choices):
    JSONChoices = json.loads(str(choices))
    text = JSONChoices["text"]

    with open('file_structures.json', 'w') as file:
        file.write(text)

    return text


def create_files_and_add_contents(filesArrayJson):
    for file in json.loads(filesArrayJson):
        print(colored(f"Creating file '{file['file_name']}'...", "cyan"))
        create_file(file["file_name"], file["file_contents"])


def create_file(file_name, file_contents):
    # Check if output directory exists, if not create one
    if not os.path.exists("output"):
        os.makedirs("output")

    file_path = os.path.join("output", file_name)

    try:
        with open(file_path, 'w') as file:
            file.write(file_contents)
        success(f"File '{file_name}' created successfully.")
    except Exception as e:
        error(f"Error creating file '{file_name}': {str(e)}")
