import click
import re
from openai import OpenAI
import os
from datetime import datetime

def is_readable(file_path):
    """Check if a file is readable as text."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file.read(1024)  # Read only the first 1024 bytes for efficiency
            return True
    except (UnicodeDecodeError, IOError):
        return False

def read_directory(path, prefix="", md_file_name="directory_contents"):
    """Recursively read the contents of a directory and write them to a Markdown file in the .aether directory."""
    aether_dir = os.path.join(os.getcwd(), '.aether')
    if not os.path.exists(aether_dir):
        os.makedirs(aether_dir)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_file_name_with_timestamp = f"{md_file_name}_{timestamp}.md"
    markdown_file_path = os.path.join(aether_dir, md_file_name_with_timestamp)

    contents = ""
    with open(markdown_file_path, 'a', encoding='utf-8', errors='replace') as md_file:
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                dir_line = f"{prefix}/{item}/\n"
                contents += dir_line
                md_file.write(f"## {dir_line}\n")
                contents += read_directory(full_path, prefix=prefix + "/" + item, md_file_name=md_file_name)
            else:
                file_line = f"{prefix}/{item}: "
                if is_readable(full_path):
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as file:
                        file_content = file.read()
                    file_line += f"\n```\n{file_content}\n```\n"
                else:
                    file_line += "[non-readable or binary content]\n"
                contents += file_line
                md_file.write(file_line)
    return contents

# Instantiate the OpenAI client
client = OpenAI()

def send_message(message_log):
    # Use the new chat completions API
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # Keep the model as is
        messages=message_log,
        max_tokens=1500,
        temperature=0.7,
    )

    # Adjusted response handling
    return response.choices[0].message.content if response.choices else ""


@click.command()
@click.argument('file_paths', nargs=-1)  # Accepts multiple file paths
def aether_inquiry(file_paths):
    """ 'ai' - Call upon the arcane intellect of an artificial intelligence to answer your questions and generate spells or Python scripts."""

    message_log = [
        {"role": "system", "content": "You are a wizard trained in the arcane. You have deep knowledge of software development and computer science. You can cast spells and read tomes to gain knowledge about problems. Please greet the user. All code and commands should be in code blocks in order to properly help the user craft spells."}
    ]

    message_log = [
            {"role": "system", "content": "You are a wizard trained in the arcane. You have deep knowledge of software development and computer science. You can cast spells and read tomes to gain knowledge about problems. Please greet the user. All code and commands should be in code blocks in order to properly help the user craft spells."}
        ]

    # Check if any file paths are provided
    if file_paths:
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # Ask user if they want to transcribe directory contents
                transcribe_confirm = input(f"Do you want to transcribe the contents of the directory '{file_path}' to the .aether directory? (yes/no): ")
                if transcribe_confirm.lower() in ['yes', 'y']:
                    # If it's a directory and user confirms, read its contents
                    directory_contents = read_directory(file_path)
                    message_log.append({"role": "user", "content": directory_contents})
                else:
                    print(f"Skipping transcription of '{file_path}'.")
            else:
                # Process files as before
                with open(file_path, 'r') as file:
                    file_content = file.read()
                message_log.append({"role": "user", "content": file_content})
        print("You provided files/folders as offerings to the aether. You may now ask your questions regarding them.")
    else:
        print("No file or folder provided. You may ask your questions to the aether.")


    last_response = ""

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("I await your summons.")
            break

        elif user_input.lower() == "scribe":
            save_prompt = input("Do you want to save the last response as a spell file, bash file, Python script, Markdown file, or just copy the last message? (spell/bash/python/markdown/copy/none): ")

            if save_prompt.lower() == "markdown":
                # Save as Markdown file
                markdown_file_name = input("Enter the name for the Markdown file (without the .md extension): ")
                with open(f"{markdown_file_name}.md", 'w') as md_file:
                    md_file.write(f"# Response\n\n{last_response}")
                print(f"Markdown saved as {markdown_file_name}.md.")

            if save_prompt.lower() == "spell":
                # Save as spell file
                code_blocks = re.findall(r'(```bash|`)(.*?)(```|`)', last_response, re.DOTALL)
                code = '\n'.join(block[1].strip() for block in code_blocks)
                spell_file_name = input("Enter the name for the spell file (without the .spell extension): ")
                spell_file_path = f".tome/{spell_file_name}.spell"
                with open(spell_file_path, 'w') as f:
                    if code_blocks:
                        f.write(code)
                    else:
                        f.write(last_response)
                print(f"Spell saved as {spell_file_name}.spell in .tome directory.")

            elif save_prompt.lower() == "bash":
                # Save as bash file
                code_blocks = re.findall(r'(```bash|`)(.*?)(```|`)', last_response, re.DOTALL)
                code = '\n'.join(block[1].strip() for block in code_blocks)
                bash_file_name = input("Enter the name for the Bash script (without the .sh extension): ")
                with open(f"{bash_file_name}.sh", 'w') as f:
                    if code_blocks:
                        f.write(code)
                    else:
                        f.write(last_response)
                print(f"Bash script saved as {bash_file_name}.sh.")

            elif save_prompt.lower() == "python":
                # Save as Python script
                code_blocks = re.findall(r'(```python|`)(.*?)(```|`)', last_response, re.DOTALL)
                code = '\n'.join(block[1].strip() for block in code_blocks)
                python_file_name = input("Enter the name for the Python script (without the .py extension): ")
                with open(f"{python_file_name}.py", 'w') as f:
                    if code_blocks:
                        f.write(code)
                    else:
                        f.write(last_response)
                print(f"Python script saved as {python_file_name}.py.")

            elif save_prompt.lower() == "copy":
                # Copy the last message
                code = last_response
                message_file_name = input("Enter the name for the message file (without the .txt extension): ")
                with open(f"{message_file_name}.txt", 'w') as f:
                    f.write(code)
                print(f"Message saved as {message_file_name}.txt.")
        else:
            message_log.append({"role": "user", "content": user_input})
            print("Querying the aether...")
            response = send_message(message_log)
            message_log.append({"role": "assistant", "content": response})
            print(f"mAGI: {response}")
            last_response = response

alias = "ai"

if __name__ == '__main__':
    aether_inquiry()