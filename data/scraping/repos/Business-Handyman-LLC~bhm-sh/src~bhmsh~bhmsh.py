import sys
import errno
import os
import subprocess
import argparse
import json

import openai

from datetime import datetime

log_file_path = None
log_session_count = 0

openai_model = 'gpt-3.5-turbo'

args = None

def openai_init():
    env_openai_api_key = None

    global args

    if args == None:
        print("ERROR :: openai_init() called without argparse initialization!")
        sys.exit(1)

    if args.token != None:
        openai.api_key = args.token
        return

    try:
        env_openai_api_key = os.environ['OPENAI_API_KEY']

    except KeyError:
        print("ERROR :: Unable to locate an OpenAI API key!")
        print("    > Explicitly supply one using the '--token' option")
        print(("    > Or add 'export OPENAI_API_KEY=<api-key-value>' to your "
               ".bash_profile (or current shell equivalent)"))
        #print((
        #    "ERROR :: Add a new line consisting of 'export "
        #    "OPENAI_API_KEY=<api-key-value>' to your '.bash_profile'!"
        #))
        #print("    > Located in your home directory")
        #print("    > May be '.bashrc' or another name depending on your system")
        sys.exit(1)

    openai.api_key = env_openai_api_key

def log_new_session():
    global log_file_path

    if log_file_path == None:
        print('ERROR :: Called log_new_session() without initializing!')
        return

    global log_session_count
    log_session_count += 1

    log_file = open(log_file_path, 'a')
    log_file.write(f'{os.linesep}')
    log_file.write(f'    ::SESSION-{log_session_count}::{os.linesep}')
    log_file.write(f'{os.linesep}')
    log_file.close()

def log_init():
    datetime_string = str(datetime.now())
    datetime_string = datetime_string.replace(' ', '-')

    global log_file_path
    log_file_path = f"{os.environ['PWD']}/{datetime_string}-gptsh.log"

    global log_session_count
    log_session_count += 1

    log_file = open(log_file_path, 'w')
    log_file.write(f'{os.linesep}')
    log_file.write(f'    ::SESSION-{log_session_count}::{os.linesep}')
    log_file.write(f'{os.linesep}')
    log_file.close()

def log_info(info=None):
    if info == None:
        print(' WARN :: No info provided to log_info()!')
        return

    if type(info) is not str:
        print(' WARN :: Passed non-string info to log_info()!')
        return

    global log_file_path

    if log_file_path == None:
        print('ERROR :: Called log_info() without initializing!')
        return

    log_file = open(log_file_path, 'a')
    log_file.write(f'{info}{os.linesep}')
    log_file.close()

def init_prompt_list():
    config_file_path = '/etc/business-handyman.d/actuator.conf'

    # If there's a .conf, use it
    if os.path.isfile(config_file_path):
        print((f" >> Using '{config_file_path}' for prompt initialization"
               f"{os.linesep}"))

        prompts = [ ]

        config_file = open(config_file_path, 'r')

        for init_prompt in config_file:
            prompts.append({
              "role": "user",
              "content": init_prompt.strip()
            })

        config_file.close()

        return prompts

    # And if there isn't a .conf, use these prompts
    prompts = [{
      "role": "user",
      "content": ("I am a Python 3 program that interfaces with ChatGPT and "
                  "serve as a bridge between an enduser and the ChatGPT model.")
    }, {
      "role": "user",
      "content": ("The enduser cannot interact with ChatGPT directly and must "
                  "go through me.")
    }, {
      "role": "user",
      "content": ("Responses from ChatGPT must be in plain text format and "
                  'must either be a Bash script with "BASH-SCRIPT:" prepended '
                  "to it or a processed version of the output from the Bash "
                  "script that ChatGPT provides. There must never be any other "
                  "type of output, including introductory or concluding "
                  "remarks.")
    }, {
      "role": "user",
      "content": ("If it's time for you to provide me with a Bash script to "
                  'run, prepend the response with "BASH-SCRIPT:" followed by '
                  "the actual Bash script. Only supply a valid bash script "
                  'with "BASH-SCRIPT:" prepended to it and no other text.')
    }, {
      "role": "user",
      "content": ("Once I execute ChatGPT's provided Bash script, I will "
                  "return its output.")
    }, {
      "role": "user",
      "content": 'The utility "curl" is installed, allowing internet access.'
    }, {
      "role": "user",
      "content": "Assume I do not have sudo shell privileges: don't use sudo."
    }, {
      "role": "user",
      "content": ("If you think sudo is manditory to accomplish the user's "
                  'request, send me "SUDO-REQUIRED" as the processed output '
                  "and I will handle it.")
    }, {
      "role": "user",
      "content": ("Interactions must be clear and concise, strictly adhering "
                  "to the given instructions, with a strong focus on providing "
                  'only "BASH-SCRIPT:" commands or processed output. No '
                  "introductory or concluding comments, including statements "
                  "like \"Understood.\" or \"I'm ready for your prompts.\" "
                  "should be made.")
    }, {
      "role": "user",
      "content": ("Do not make assumptions about the system state or add "
                  "unnecessary commentary to responses.")
    }, {
      "role": "user",
      "content": ("Maintain a focus on providing accurate information in a "
                  "precise manner. Begin.")
    }]

    return prompts

def main():
    # TODO: Inject version here
    print(f'{os.linesep}')
    print('        ~~~    BHMsh v1.1.3    ~~~')
    print(f'{os.linesep}')

    arg_parser = argparse.ArgumentParser(prog='GPTsh',
                                         description=('OpenAI ChatGPT shell '
                                                      'controller (via Bash)'))

    arg_parser.add_argument('-l',
                            '--log',
                            action='store_true',
                            help=('Log session(s) to a file. Log file '
                                  'information printed upon execution.'))
    arg_parser.add_argument('-t',
                            '--token',
                            help='Explicitly provide an OpenAI API key.')

    global args
    args = arg_parser.parse_args()

    openai_init()

    if args.log:
        log_init()
        print((f" >> Logging to \"{log_file_path}\""
               f'{os.linesep}'))

    prompts = init_prompt_list()

    if args.log:
        for prompt in prompts:
            log_info(json.dumps(prompt, indent=4))

    print(f" >> How can I help you today?{os.linesep}")

    while True:
        user_input = None

        try:
            user_input = input(' << ')
            print('')

        except EOFError:
            print(f'{os.linesep}')
            print(f' >> BHMsh terminating...{os.linesep}')

            sys.exit()

        if len(user_input) == 0:
            print(f' >> No user input supplied... Try again!{os.linesep}')
            continue

        if user_input == 'h' or user_input == 'help':
            print(f' >> Available commands:{os.linesep}')
            print(("    (c)lear        - Clears the current ChatGPT session "
                   "and initializes a new one."))
            print(f"    (q)uit, (e)xit - Exit BHMsh{os.linesep}")
            continue

        elif user_input == 'c' or user_input == 'clear':
            prompts = init_prompt_list()

        elif user_input == 'q' or user_input == 'quit':
            print(f'{os.linesep}')
            print(f' >> GPTsh terminating...{os.linesep}')

            sys.exit()

        elif user_input == 'e' or user_input == 'exit':
            print(f'{os.linesep}')
            print(f' >> GPTsh terminating...{os.linesep}')

            sys.exit()

            if args.log:
                log_new_session()

            continue

        user_input_dict = {
            'role': 'user',
            'content': user_input
        }

        prompts.append(user_input_dict)

        if args.log:
            log_info(f'{json.dumps(user_input_dict, indent=4)}')

        chat_completion = None

        # Break out of here when ChatGPT responds with a message
        # without the prepended sentinel 'BASH-SCRIPT:'
        while True:
            chat_completion = openai.ChatCompletion.create(model=openai_model,
                                                           messages=prompts)

            chatgpt_res = chat_completion.choices[0].message.content

            chatgpt_res_dict = {
                'role': 'assistant',
                'content': chatgpt_res
            }

            prompts.append(chatgpt_res_dict)

            if args.log:
                log_info(f'{json.dumps(chatgpt_res_dict, indent=4)}')

            sentinel_tag_val = 'BASH-SCRIPT:'
            sentinel_tag_idx = -1

            sentinel_tag_idx = chatgpt_res.find(sentinel_tag_val)

            # No sentinel 'BASH-SCRIPT:' included in response
            #    > It's ChatGPT's processed output
            if sentinel_tag_idx < 0:
                break

            # TODO: Make sure there's something after 'BASH-SCRIPT:'

            bash_script = chatgpt_res[sentinel_tag_idx+len(sentinel_tag_val):]

            bash_script += '\necho "::CWD::$PWD"'

            # TODO: file error handling here

            tmp_script = '.bhmsh.tmp.sh'

            temp_bash_script_f = open(tmp_script, 'w')
            temp_bash_script_f.write(bash_script)
            temp_bash_script_f.close()

            try:
                myProc = subprocess.run(args = [ '/bin/bash', tmp_script ],
                                        capture_output = True)

            finally:
                os.remove('.bhmsh.tmp.sh')

            bash_output = None

            try:
                myProc.check_returncode()

            except subprocess.CalledProcessError:
                bash_output = myProc.stderr.decode('utf-8')

            else:
                bash_output = myProc.stdout.decode('utf-8')

                new_cwd_idx = bash_output.find('::CWD::')

                if new_cwd_idx >= 0:
                    new_cwd = bash_output[new_cwd_idx+len('::CWD::'):]
                    new_cwd = new_cwd.strip()

                    os.chdir(new_cwd)

                    bash_output = bash_output[:new_cwd_idx]

                if len(bash_output) <= 0:
                    bash_output = 'COMPLETED'

            gptsh_res_dict = {
                'role': 'user',
                'content': bash_output
            }

            # TODO: Send "too long" sentinel to ChatGPT

            prompts.append(gptsh_res_dict)

            if args.log:
                log_info(f'{json.dumps(gptsh_res_dict, indent=4)}')

        chatgpt_processed_res = chat_completion.choices[0].message.content

        print(f' >> {chatgpt_processed_res}\n')

if __name__ == '__main__':
    main()
