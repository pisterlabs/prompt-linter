import subprocess
import xml.etree.ElementTree as ET
from openai import OpenAI

INIT_INSTRUCTION = """You are being run in a scaffold in a shell on a Macbook. When you want to run a shell command, write it in a <bash> XML tag, for example, for the command `ls`, you should output `<bash>ls</bash>`. You will be shown the result of the command and be able to run more commands. Other things you say will be sent to the user. In cases where you know how to do something, don't explain how to do it, just start doing it by emitting bash commands one at a time. The user uses fish, but you're in a bash shell. Remember that you can't interact with stdin directly, so if you want to e.g. do things over ssh you need to run commands that will finish and return control to you rather than blocking on stdin. Don't wait for the user to say okay before suggesting a bash command to run. If possible, don't include an explanation, just say the command.

If you can't do something without assistance, please suggest a way of doing it without assistance anyway."""

COMMAND_COLOR = "\033[31m" # red
ASSISTANT_COLOR = "\033[0;34m" # blue
SYSTEM_COLOR = "\033[0;32m" # green
RESET = "\033[0m" # black

def cprint(message, color, end='\n'):
    print(f"{color}{message}{RESET}", end=end)

def chat():

    client = OpenAI()

    chat_history = [
        {"role": "system", "content": INIT_INSTRUCTION}
    ]
    feedback = ""
    cprint("System: " + INIT_INSTRUCTION, SYSTEM_COLOR)

    while True:
        
        print("\n")

        if feedback:
            chat_history.append({"role": "system", "content": feedback})
            feedback = "" 
        else:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            chat_history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=chat_history,
                stream=True,
            )
            response_text = ""
            cprint("Assistant: ", ASSISTANT_COLOR, end='')
            for event in response:
                event_text = event.choices[0].delta.content  # extract the text
                if event_text is not None:
                    cprint(event_text, ASSISTANT_COLOR, end='')  # print the text
                    response_text += event_text  # append the text
            print("\n")

            chat_history.append({"role": "assistant", "content": response_text})
    
            # Check for <bash> tag and execute command
            if '<bash>' in response_text:
                try:
                    root = ET.fromstring(f'<root>{response_text}</root>')
                    for command in root.findall('bash'):
                        cmd = command.text.strip()
                        
                        confirmation = input(f"Press Enter to run the command {COMMAND_COLOR}`{cmd}`{RESET}, or type your feedback: ")
                        
                        if confirmation == '':
                            # TODO: try and then except subprocess.CalledProcessError as e:
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
                            command_output = f"<result>{result.stdout}</result>" if result.stdout else "No output"
                            system_message = f"Executed: {cmd}\nOutput: {command_output}"
                            cprint("System:\n" + system_message, SYSTEM_COLOR)
                            chat_history.append({"role": "system", "content": system_message})
                        else:
                            feedback = confirmation # will be fed into the model in the next iteration
                except Exception as e:
                    print(f"Error executing command: {e}")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    chat()