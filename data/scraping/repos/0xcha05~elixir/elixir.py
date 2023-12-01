import difflib
import json
import os
import shutil
import subprocess
import sys

import openai
from termcolor import cprint

# Set up the OpenAI API
with open("openai_key.txt") as f:
    openai.api_key = f.read().strip()

# Mapping of file extensions to Docker images
DOCKER_IMAGES = {
    ".py": "python:3.8",
    ".js": "node:14",
}


def run_script(script_name, *args):
    current_directory = os.getcwd()
    file_ext = os.path.splitext(script_name)[-1]
    docker_image = DOCKER_IMAGES.get(file_ext)

    if not docker_image:
        print(f"Unsupported file extension: {file_ext}")
        sys.exit(1)

    container_name = "elixir-container"

    # Check if container is already running, if not start it
    if (
        container_name
        not in subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
        ).stdout
    ):
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "-v",
                f"{current_directory}:/app",
                "--name",
                container_name,
                docker_image,
                "sleep",
                "infinity",
            ]
        )

    docker_command = f"docker exec {container_name} {docker_image.split(':')[0]} /app/{script_name} {' '.join(args)}"

    docker_process = subprocess.Popen(
        docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

    docker_process.wait()  # Wait for the subprocess to finish

    # Print the logs from the Docker container
    while True:
        output = docker_process.stdout.readline()
        if not output:
            break
        print(output.decode("utf-8").strip())

    return docker_process.stdout.read().decode("utf-8"), docker_process.returncode


def send_error_to_gpt4(file_path, args, error_message):
    with open(file_path, "r") as f:
        file_lines = f.readlines()

    file_with_lines = []
    for i, line in enumerate(file_lines):
        file_with_lines.append(str(i + 1) + ": " + line)
    file_with_lines = "".join(file_with_lines)

    with open("prompt.txt") as f:
        initial_prompt_text = f.read()

    prompt = (
        initial_prompt_text + "\n\n"
        "Here is the script that needs fixing:\n\n"
        f"{file_with_lines}\n\n"
        "Here are the arguments it was provided:\n\n"
        f"{args}\n\n"
        "Here is the error message:\n\n"
        f"{error_message}\n"
        "Please provide your suggested changes, and remember to stick to the "
        "exact format as described above."
    )

    # print(prompt)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=1.0,
    )

    return response.choices[0].message.content.strip()


def apply_changes(file_path, changes_json):
    changes = json.loads(changes_json)

    # Filter out explanation elements
    code = changes["script"]
    explanations = changes["explanation"]

    with open(file_path, "w") as f:
        f.writelines(code)

    # Print explanations
    cprint("Explanations:", "blue")
    cprint(f"- {explanations}", "blue")


def main():
    if len(sys.argv) < 1:
        print("Usage: elixir.py <script_name> ... [--revert]")
        sys.exit(1)

    script_name = sys.argv[1]
    args = sys.argv[2:]

    # Revert changes if requested
    if "--revert" in args:
        backup_file = script_name + ".bak"
        if os.path.exists(backup_file):
            shutil.copy(backup_file, script_name)
            print(f"Reverted changes to {script_name}")
            sys.exit(0)
        else:
            print(f"No backup file found for {script_name}")
            sys.exit(1)

    # Make a backup of the original script
    shutil.copy(script_name, script_name + ".bak")

    while True:
        output, returncode = run_script(script_name, *args)

        if returncode == 0:
            cprint("Script ran successfully.", "blue")
            print("Output:", output)
            break
        else:
            cprint("Script crashed. Trying to fix...", "blue")
            print("Output:", output)

            json_response = send_error_to_gpt4(script_name, args, output)
            apply_changes(script_name, json_response)
            cprint("Changes applied. Rerunning...", "blue")


if __name__ == "__main__":
    main()
