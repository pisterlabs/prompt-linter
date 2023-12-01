import argparse
import configparser
import openai
import os
import re
import datetime
import shutil
import time
from plyer import notification
import platform
import threading
from rich import print as rprint
from rich.markdown import Markdown
from rich.table import Table
import tiktoken
import platform
import sys
if platform.system() == "Windows":
    try:
        import pyreadline as readline
    except ImportError:
        print("Please install pyreadline using: pip install pyreadline")
        sys.exit(1)
else:
    import readline

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-16k":
        print("Warning: gpt-3.5-turbo-16k may change over time. Returning num tokens assuming gpt-3.5-turbo-16k-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model in {"gpt-4-0314", "gpt-4-0613", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-4", "gpt-3.5-turbo"}:
        tokens_per_message = 3
        tokens_per_name = 1
    
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def play_sound(sound_file_path):
    system = platform.system()
    
    if system == 'Windows':
        import winsound
        winsound.PlaySound(sound_file_path, winsound.SND_FILENAME)
    elif system == 'Darwin':
        os.system(f'afplay {sound_file_path}')
    elif system == 'Linux':
        os.system(f'aplay {sound_file_path}')
    else:
        print("Unable to play notification sound: Unsupported platform")

def send_notification(secs_taken, model_name):
    notification.notify(
        title='Request Finished!',
        message='Took {:.2f} seconds. Model: {}'.format(secs_taken, model_name),
        app_name='fpt',
        timeout=10
    )
    play_sound("notification.wav")

def process_latex(content):
    # Regex patterns for inline and non-inline LaTeX
    inline_pattern = r'\$(.+?)\$'
    non_inline_pattern = r'\$\$([\s\S]*?)\$\$'

    # Function to add a \ before every \ and wrap with inline code and $
    def replace_inline(match):
        return r'`$' + match.group(1) + r'$`'

    # Function to add a \ before every \ and wrap with code block and $$
    def replace_non_inline(match):
        return '```latex\n$$' + match.group(1) + '$$\n```'

    # Replace inline LaTeX with wrapped inline code
    content = re.sub(inline_pattern, replace_inline, content)

    # Replace non-inline LaTeX with wrapped code block
    content = re.sub(non_inline_pattern, replace_non_inline, content)

    return content

def render_markdown_with_tables(markdown_string):
    # Process LaTeX in the input string
    markdown_string = process_latex(markdown_string)

    # Split input into lines
    lines = markdown_string.strip().split('\n')

    # Initialize variables
    in_table = False
    current_table_data = []
    current_table_header = []
    current_table_justifications = []
    non_table_content = []

    # Iterate through lines
    for line in lines:
        if re.match(r'\|\s*[^\s]', line) and not in_table:  # Check if line starts a table
            in_table = True
            
            # Print previous non-table content
            if non_table_content:
                print(Markdown('\n'.join(non_table_content)))
                non_table_content = []

            header_line = re.split(r'\s*\|\s*', line.strip())[1:-1]
            current_table_header = header_line
        elif in_table and re.match(r'\|\s*:?-+:?\s*\|', line) and not current_table_justifications:  # Check if line is a table separator
            justifications_line = re.split(r'\s*\|\s*', line.strip())[1:-1]
            current_table_justifications = [get_justification(j) for j in justifications_line]
            if len(current_table_header) != len(current_table_justifications):
                current_table_justifications = ['left'] * len(current_table_header)  # Default to left alignment if not specified
        elif in_table and re.match(r'\|', line):  # Check if line is a table row
            row_line = re.split(r'\s*\|\s*', line.strip())[1:-1]
            current_table_data.append(row_line)
        else:
            # Save non-table content
            non_table_content.append(line)

            # Print table if it exists
            if in_table:
                table = Table()
                for idx, header in enumerate(current_table_header):
                    table.add_column(header, justify=current_table_justifications[idx])

                for row_data in current_table_data:
                    table.add_row(*row_data)

                print(table)
                # Reset table variables
                in_table = False
                current_table_data = []
                current_table_header = []
                current_table_justifications = []

    # Print the last table if it exists
    if in_table:
        table = Table()
        for idx, header in enumerate(current_table_header):
            table.add_column(header, justify=current_table_justifications[idx])

        for row_data in current_table_data:
            table.add_row(*row_data)

        print(table)

    # Print remaining non-table content
    if non_table_content:
        print(Markdown('\n'.join(non_table_content)))

def get_justification(j_line):
    if j_line.startswith(':') and j_line.endswith(':'):
        return 'center'
    elif j_line.endswith(':'):
        return 'right'
    else:
        return 'left'

# convert a list of strings into messages, including custom prompts
def construct_messages_from_sections(sections):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "The following messages you will receive contain **Markdown formatting**, and you can reply to them using Markdown formatting, like links, tables, bold, italics, code blocks, inline latex, etc. You can highlight the key words with **bold**, render the math formulas using inline $euqations$, and use markdown tables to show suitable information. Reply with 'understood' to continue."},
        {"role": "assistant", "content": "Understood. I'm ready to proceed with your questions and messages in Markdown formatting."},
        ]
    for i, section in enumerate(sections):
        if i % 2 == 0:
            messages.append({"role": "user", "content": section})
        else:
            messages.append({"role": "assistant", "content": section})
    return messages

def sendToGPT(sections, is_gpt_4, fail_save=False):
    global archive_directory
    messages = construct_messages_from_sections(sections)
    target_file = generate_filename(archive_directory)
    target_file = os.path.join(archive_directory, target_file)
    try:
        return GPTRequest(messages, is_gpt_4)
    except (openai.error.Timeout, openai.error.APIError, openai.error.APIConnectionError, KeyboardInterrupt) as e:
        print(e)
        if type(e) == KeyboardInterrupt:
            print("Gracefully exiting...")
        if fail_save and len(sections) > 2:
            write_sections_to_file(sections, target_file)
            rprint("There was an error during the last request. Saved the unfinished thread to {} as backup.".format(target_file))
        exit()

# send messages to GPT and return the response
def GPTRequest(messages, is_gpt_4):
    global args, gpt_4_model, gpt_3_5_model
    if is_gpt_4:
        model = gpt_4_model
        price_rate_input = 0.03
        price_rate_output = 0.06
    else:
        model = gpt_3_5_model
        price_rate_input = 0.0015
        price_rate_output = 0.002
    start_time = time.time()
    if args.verbose:
        rprint("Verbose: sending request using {}, messages = {}".format(model, messages))
    response = openai.ChatCompletion.create(model = model, messages = messages)
    text = response["choices"][0]["message"]["content"]
    prompt_tokens = response["usage"]["prompt_tokens"]
    completion_tokens = response["usage"]["completion_tokens"]
    total_tokens = response["usage"]["total_tokens"]
    end_time = time.time()
    spent_cents = (prompt_tokens * price_rate_input + completion_tokens * price_rate_output) / 10
    if args.verbose or config.getboolean('Options', 'show_tokens'):
        rprint(f"[dim]\[fpt] Request finished. Model: [bold cyan]{model}[/bold cyan], api_base: [bold cyan]{openai.api_base}[/bold cyan], took [bold cyan]{end_time - start_time:.2f}[/bold cyan] seconds. Used tokens: [bold cyan]{total_tokens}[/bold cyan] ([bold cyan]{prompt_tokens}[/bold cyan] prompt + [bold cyan]{completion_tokens}[/bold cyan] response). Calculated cost: [bold cyan]{spent_cents:.2f}[/bold cyan] cents[/dim]")
    if config.getboolean('Options', 'notifications'):
        notification_thread = threading.Thread(target=send_notification, args=(end_time - start_time, model))
        notification_thread.start()
    return text, prompt_tokens, completion_tokens, total_tokens

def stream_to_stdout_or_file(sections, is_gpt_4, file=None):
    global gpt_3_5_model, gpt_4_model
    if is_gpt_4:
        model = gpt_4_model
        price_rate_input = 0.03
        price_rate_output = 0.06
    else:
        model = gpt_3_5_model
        price_rate_input = 0.0015
        price_rate_output = 0.002
    messages = construct_messages_from_sections(sections)
    prompt_tokens = num_tokens_from_messages(messages, model=model)
    response_text = ""
    if file:
        reformat_end_of_file(file)

    start_time = time.time()
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        stream = True,
    )
    for chunk in response:
        if chunk["choices"][0]["finish_reason"] == "stop":
            if not file:
                print("")
            break
        if "content" not in chunk["choices"][0]["delta"]:
            continue
        chunk_text = chunk["choices"][0]["delta"]["content"]
        if file:
            with open(file, 'a') as f:
                f.write(chunk_text)
        else:
            print(chunk_text, end="")
        response_text += chunk_text

    messages.append({"role": "assistant", "content": response_text})
    total_tokens = num_tokens_from_messages(messages, model=model) - 4
    completion_tokens = total_tokens - prompt_tokens
    spent_cents = (prompt_tokens * price_rate_input + completion_tokens * price_rate_output) / 10
    end_time = time.time()
    if args.verbose or config.getboolean('Options', 'show_tokens'):
        rprint(f"[dim]\[fpt] Stream request finished. Model: [bold cyan]{model}[/bold cyan], api_base: [bold cyan]{openai.api_base}[/bold cyan], took [bold cyan]{end_time - start_time:.2f}[/bold cyan] seconds. Used tokens: [bold cyan]{total_tokens}[/bold cyan] ([bold cyan]{prompt_tokens}[/bold cyan] prompt + [bold cyan]{completion_tokens}[/bold cyan] response). Calculated cost: [bold cyan]{spent_cents:.2f}[/bold cyan] cents[/dim]")
    if file:
        reformat_end_of_file(file)
    return response_text

# function to insert "> " in front of each line in a string (markdown blockquote)
def insert_gt(string):
    lines = string.splitlines()
    for i in range(len(lines)):
        lines[i] = "> " + lines[i]
    return "\n".join(lines)

# check if a string is a markdown blockquote
def is_md_blockquote(s):
    for i, line in enumerate(s.split('\n')):
        # also accept obsidian callout blockquote
        if i == 0 and line == '> [!question]':
            continue
        if not line.startswith('> '):
            return False
    return True

# delete the first two characters of each line in a string
def delete_first_two_chars(string):
    lines = string.split('\n')
    for i in range(len(lines)):
        lines[i] = lines[i][2:]
    new_string = '\n'.join(lines)
    return new_string

# remove the markdown blockquote formatting from a string
def remove_md_blockquote_if_present(string):
    if is_md_blockquote(string):
        return delete_first_two_chars(string)
    else:
        return string

# add markdown blockquote formatting to a string
def add_md_blockquote_if_not_present(string):
    global args
    ob_enabled = args.obsidian
    if is_md_blockquote(string):
        return string
    else:
        return insert_gt(string) if not ob_enabled else '> [!question]\n' + insert_gt(string)

# handle the formatting at end of the file
def reformat_end_of_file(file):
    with open(file, 'r') as f:
        content = f.read()
    if content.isspace() or content == '':
        content = ''
        with open(file, 'w') as f:
            f.write(content)
    elif re.search(r'\n\n----\n*$', content):
        content = re.sub(r'\n\n----\n*$', '\n\n----\n\n', content)
        with open(file, 'w') as f:
            f.write(content)
    else:
        content = re.sub(r'\n*$', '\n\n----\n\n', content, 1)
        with open(file, 'w') as f:
            f.write(content)

# append a message to the end of a file
def append_message_to_file(message, file, type):
    # type is either 'prompt' or 'response'
    if type == 'prompt':
        message = insert_gt(message)
    elif type != 'response':
        print('Error: type must be either prompt or response when using append_message_to_file.')
        exit()
    reformat_end_of_file(file)
    # append the message to the end of the file
    with open(file, 'a') as f:
        f.write(message + '\n\n----\n\n')

# remove the last message from a file
def remove_last_message_from_file(file):
    with open(file, 'r') as f:
        content = f.read()
    sections = content.split('\n\n----\n\n')
    if len(sections) <= 2 or sections[-1] != '':
        print("Something's wrong at remove_last_message_from_file. You win! Open an issue on GitHub.")
    else:
        sections = sections[:-2]
        content = '\n\n----\n\n'.join(sections)
        with open(file, 'w') as f:
            f.write(content + '\n\n----\n\n')

def write_sections_to_file(sections, file):
    for i, section in enumerate(sections):
        if i % 2 == 0:
            section = add_md_blockquote_if_not_present(section)
        else:
            section = remove_md_blockquote_if_present(section)
        sections[i] = section
    content = '\n\n----\n\n'.join(sections)
    with open(file, 'w') as f:
        f.write(content)

# assuming the format is correct, and the last message is a prompt, blockquote the last message
def blockquote_last_message(file):
    with open(file, 'r') as f:
        content = f.read()
    sections = content.split('\n\n----\n\n')
    if len(sections) == 1 or sections[-1] != '':
        print("Something's wrong at blockquote_last_message. You win! Open an issue on GitHub.")
    else:
        sections[-2] = add_md_blockquote_if_not_present(sections[-2])
        content = '\n\n----\n\n'.join(sections)
        with open(file, 'w') as f:
            f.write(content)

# for new files, blockquote the prompt and add a horizontal rule
def blockquote_file(file):
    with open(file, 'r') as f:
        content = f.read()
    content = add_md_blockquote_if_not_present(content)
    with open(file, 'w') as f:
        f.write(content + '\n\n----\n\n')

def append_to_file(file, content):
    with open(file, 'a') as f:
        f.write(content)

def prepend_to_file(file, content):
    with open(file, 'r+') as f:
        original_content = f.read()
        f.seek(0, 0)
        f.write(content + original_content)

# returns a tuple (type, messages), mixed use for optimization
# 1. check if a file is a previously saved thread, or conforms to the format of a saved thread
# types can be "plain", "vaild_ends_with_prompt", "valid_ends_with_response", "invalid_ordering"
# 2. if the file is valid, return the list of messages, otherwise return an empty list
# also reformats the file
def file_type_check_get_messages(file):
    reformat_end_of_file(file)
    with open(file, 'r') as f:
        content = f.read()
    if re.search(r'\n\n----\n\n', content):
        sections = content.split('\n\n----\n\n')
        if sections[-1] == '' or sections[-1].isspace():
            sections.pop()
        if len(sections) == 1:
            blockquote_last_message(file)
            return "plain", [remove_md_blockquote_if_present(sections[0])]
        for i, section in enumerate(sections[:-1]):
            if i % 2 == 0:
                if not is_md_blockquote(section):
                    return "invalid_ordering", []
                sections[i] = delete_first_two_chars(section)
            else:
                if is_md_blockquote(section):
                    return "invalid_ordering", []
        if len(sections) % 2 == 0:
            if is_md_blockquote(sections[-1]):
                return "invalid_ordering", []
            else:
                return "valid_ends_with_response", sections
        else:
            sections[-1] = remove_md_blockquote_if_present(sections[-1])
            blockquote_last_message(file)
            return "valid_ends_with_prompt", sections
    else:
        if content == '' or content.isspace():
            return "empty", []
        blockquote_file(file)
        return "plain", [remove_md_blockquote_if_present(content)]

# generate a filename for a new thread
def generate_filename(archive_directory):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    files = os.listdir(archive_directory)
    todays_files = [file for file in files if file.startswith(today)]
    used_numbers = [int(file.split('-')[-1].split('.')[0][2:]) for file in todays_files]
    # Find the first available sequential number up to three digits
    for number in range(1, 1000):
        if number not in used_numbers:
            seq_number = f"{number:02d}"
            break
    new_filename = f"{today}-ai{seq_number}.md"
    return new_filename

def format_headless_thread_content(sections):
    content = ''
    for i, section in enumerate(sections):
        if i % 2 == 0:
            content += add_md_blockquote_if_not_present(section) + '\n\n----\n\n'
        else:
            content += section + '\n\n----\n\n'
    return content

# headless mode
def headless_mode():
    global usage_history_file
    global args
    global prepend_history
    global stream
    sections = []
    target_file = generate_filename(archive_directory)
    target_file = os.path.join(archive_directory, target_file)
    config_file = os.path.join(os.getcwd(), 'fpt.conf')
    rprint('Welcome to fpt! Enter your question after the > and hit enter to continue.\nEnter q to save thread to history and exit. Enter qf to save thread to a seperate file and exit.\nHistory location: {}\nqf will save to: {}\nYou can change your settings at: {}'.format(usage_history_file, target_file, config_file))
    while True:
        user_input = input("> ")
        if user_input == 'q':
            if len(sections) == 0:
                exit()
            content_to_write = format_headless_thread_content(sections)
            if prepend_history:
                prepend_to_file(usage_history_file, content_to_write)
            else:
                append_to_file(usage_history_file, content_to_write)
            exit()
        elif user_input == 'qf':
            if len(sections) == 0:
                exit()
            content_to_write = format_headless_thread_content(sections)
            with open(target_file, 'w') as f:
                f.write(content_to_write)
            print('Saved thread to {}'.format(target_file))
            exit()
        else:
            sections.append(user_input)
            if stream:
                response = stream_to_stdout_or_file(sections, is_gpt_4=args.gpt4)
            else:
                response, _, _, _ = sendToGPT(sections, args.gpt4, fail_save=True)
                render_markdown_with_tables(response)
            sections.append(response)

# interactive mode
def interactive_mode():
    global usage_history_file
    global args
    global stream
    while True:
        user_input = input("Type the next question or command, h for help: ")
        if user_input == 'h':
            print('f: read next question from file. f3 to force GPT-3.5, f4 to force GPT-4')
            print('r: re-generate the last response. r3 to force GPT-3.5, r4 to force GPT-4')
            print('o: read the file and respond to the last one question only. o3 to force GPT-3.5, o4 to force GPT-4.')
            print('d: dump-to-history (clear the current file and archive the cleared thread into history file)')
            print('df: dump-to-file (clear the current file and archive the cleared thread into a new file in the archive directory, with a generated file name)')
            print('q: quit the program')
            print('h: print this help message')
        elif user_input == 'f' or user_input == 'f3' or user_input == 'f4':
            if user_input == 'f3':
                is_gpt_4 = False
            elif user_input == 'f4':
                is_gpt_4 = True
            else:
                is_gpt_4 = args.gpt4
            type, sections = file_type_check_get_messages(args.file)
            if type == "empty":
                print('The file is empty. Please type your question.')
            elif type == "invalid_ordering":
                print('Invalid ordering in the file. Please check the file and try again.')
            elif type == "valid_ends_with_response":
                print('The file ends with a response. Please ask a question at the end of thread.')
            else:
                if stream:
                    stream_to_stdout_or_file(sections, is_gpt_4, args.file)
                else:
                    response, _, _, _ = sendToGPT(sections, is_gpt_4)
                    append_message_to_file(response, args.file, 'response')
        elif user_input == 'r' or user_input == 'r3' or user_input == 'r4':
            if user_input == 'r3':
                is_gpt_4 = False
            elif user_input == 'r4':
                is_gpt_4 = True
            else:
                is_gpt_4 = args.gpt4
            remove_last_message_from_file(args.file)
            type, sections = file_type_check_get_messages(args.file)
            if type == "invalid_ordering":
                print('Invalid ordering in the file. Please check the file and try again.')
            elif type == "valid_ends_with_response":
                print('The file ends with a response. This shouldn\'t happen...')
            else:
                if stream:
                    stream_to_stdout_or_file(sections, is_gpt_4, args.file)
                else:
                    response, _, _, _ = sendToGPT(sections, is_gpt_4)
                    append_message_to_file(response, args.file, 'response')
        elif user_input == 'o' or user_input == 'o3' or user_input == 'o4':
            if user_input == 'o3':
                is_gpt_4 = False
            elif user_input == 'o4':
                is_gpt_4 = True
            else:
                is_gpt_4 = args.gpt4
            type, sections = file_type_check_get_messages(args.file)
            if type == "invalid_ordering":
                print('Invalid ordering in the file. Please check the file and try again.')
            elif type == "valid_ends_with_response":
                print('The file ends with a response. Please ask a question at the end of thread.')
            else:
                if stream:
                    stream_to_stdout_or_file(sections, is_gpt_4, args.file)
                else:
                    response, _, _, _ = sendToGPT([sections[-1]], is_gpt_4)
                    append_message_to_file(response, args.file, 'response')
        elif user_input == 'd' or user_input == 'df':
            if user_input == 'd':
                target_file = usage_history_file
                with open(args.file, 'r') as f:
                    content = f.read()
                if prepend_history:
                    prepend_to_file(target_file, content)
                else:
                    append_to_file(target_file, content)
            else:
                target_file = generate_filename(archive_directory)
                target_file = os.path.join(archive_directory, target_file)
                shutil.copy(args.file, target_file)
            with open(args.file, 'w') as f:
                f.write('')
            print('Cleared the current file and archived the cleared thread to {}'.format(target_file))
        elif user_input == 'q':
            exit()
        else:
            type, sections = file_type_check_get_messages(args.file)
            if type == "invalid_ordering":
                print('Invalid ordering in the file. Please check the file and try again.')
            elif type == "valid_ends_with_prompt":
                print('The file ends with a prompt. Please use \'f\' if you want to continue the thread.')
            elif type == "plain":
                print('No thread detected. Please use \'f\' if you want to start a thread from the content in the file.')
            else:
                append_message_to_file(user_input, args.file, 'prompt')
                sections.append(user_input)
                if stream:
                    stream_to_stdout_or_file(sections, args.gpt4, args.file)
                else:
                    response, _, _, _ = sendToGPT(sections, args.gpt4)
                    append_message_to_file(response, args.file, 'response')

# parse the command line arguments
parser = argparse.ArgumentParser(
    prog='fpt',
    description='A CLI for OpenAI\'s GPT-3.5/GPT-4 API',
)
input_group = parser.add_mutually_exclusive_group()
input_group.add_argument('-f', '--file', type=str, help='The file to operate on')
input_group.add_argument('-q', '--question', type=str, help='A single question to send to GPT')
parser.add_argument('-4', '--gpt4', action='store_true', help='Use GPT-4')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
parser.add_argument('-o', '--obsidian', action='store_true', help='Save blockquotes as Obsidian callouts')
args = parser.parse_args()

# turn file path into absolute path
if args.file:
    args.file = os.path.abspath(args.file)

# set current working directory to the directory of this script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# read the config file
config = configparser.ConfigParser()
config.read('fpt.conf')
if config.has_option('OpenAI', 'custom_api_base'):
    openai.api_base = config.get('OpenAI', 'custom_api_base')
    api_key = config.get('OpenAI', 'custom_api_key')
else:
    api_key = config.get('OpenAI', 'api_key')
archive_directory = config.get('Directories', 'archive_directory')
usage_history_file = config.get('Directories', 'usage_history_file')
prepend_history = config.getboolean('Options', 'prepend_history')
stream = config.getboolean('Options', 'stream')
gpt_3_5_model = config.get('OpenAI', 'gpt_3_5_model')
gpt_4_model = config.get('OpenAI', 'gpt_4_model')

# set the API key
openai.api_key = api_key

# create the archive directory and usage history file if they don't exist
if not os.path.exists(archive_directory):
    os.makedirs(archive_directory)
if not os.path.exists(usage_history_file):
    open(usage_history_file, 'a').close()

# if the user asked a single question, answer it, save the response, and exit
if args.question:
    if stream:
        response = stream_to_stdout_or_file([args.question], is_gpt_4=args.gpt4)
    else:
        response, _, _, _ = sendToGPT([args.question], is_gpt_4=args.gpt4)
        render_markdown_with_tables(response)
    content_to_write = add_md_blockquote_if_not_present(args.question) + '\n\n----\n\n' + response + '\n\n----\n\n'
    if prepend_history:
        prepend_to_file(usage_history_file, content_to_write)
    else:
        append_to_file(usage_history_file, content_to_write)

# if the user asked to operate on a file, enter interactive mode
elif args.file:
    if not os.path.isfile(args.file):
        print('Error: file does not exist. Exiting...')
        exit()
    type, sections = file_type_check_get_messages(args.file)
    if type == 'empty':
        print('File is empty. Entering interactive mode...')
        interactive_mode()
    elif type == 'invalid_ordering':
        print('Error: invalid ordering of messages. Make sure there are alternating prompts and responses. Exiting...')
        exit()
    elif type == 'valid_ends_with_response':
        print('File ends with a response. Entering interactive mode...')
        interactive_mode()
    elif type == 'valid_ends_with_prompt' or type == 'plain':
        if stream:
            stream_to_stdout_or_file(sections, is_gpt_4=args.gpt4, file=args.file)
        else:
            response, _, _, _ = sendToGPT(sections, is_gpt_4=args.gpt4)
            append_message_to_file(response, args.file, 'response')
        interactive_mode()
    else:
        print('Error: invalid file type. Exiting...')
        exit()

# if the user didn't specify a file or a question, enter headless interactive mode
else:
    headless_mode()
