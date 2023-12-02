import click
import openai
import re

# Function to send a message to the OpenAI chatbot model and return its response
def send_message(message_log):
    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = openai.ChatCompletion.create(
        model="gpt-4",  # The name of the OpenAI chatbot model to use
        messages=message_log,   # The conversation history up to this point, as a list of dictionaries
        max_tokens=1500,        # The maximum number of tokens (words or subwords) in the generated response
        stop=None,              # The stopping sequence for the generated response, if any (not used here)
        temperature=0.7,        # The "creativity" of the generated response (higher temperature = more creative)
    )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    # If no response with text is found, return the first response's content (which may be empty)
    return response.choices[0].message.content

@click.command()
@click.argument('file_path', required=False)  # Adding file_path as an optional argument
def aether_inquiry(file_path=None):
    """Call upon the arcane intellect of an artificial intelligence to answer your questions and generate spells or Python scripts."""

    message_log = [
        {"role": "system", "content": "You are a wizard trained in the arcane. You have deep knowledge of software development and computer science. You can cast spells and read tomes to gain knowledge about problems. Please greet the user. All code and commands should be in code blocks in order to properly help the user craft spells."}
    ]

    # If a file path is provided, read the file and append its content to the message_log
    if file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()
        message_log.append({"role": "user", "content": file_content})
        print("You provided a file as offering to the aether. You may now ask your question regarding it.")

    last_response = ""
    first_request = False

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("I await your summons.")
            break

        elif user_input.lower() == "scribe":
            # Prompt the user whether they want to save the last response as a spell file, bash file, Python script, or just copy the last message
            save_prompt = input("Do you want to save the last response as a spell file, bash file, Python script, or just copy the last message? (spell/bash/python/copy/none): ")

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