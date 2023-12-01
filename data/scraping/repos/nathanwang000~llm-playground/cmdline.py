'''
cmdline with llm

Runs commands in following order
1. try functions defined in prompt
2. try excecute a system command
3. ask llm
'''
import openai
import click
import os, glob, re
import subprocess
from termcolor import colored
from lib.utils import ChatBot, repl, strip_multiline
from lib.utils import ShellCompleter, EXCEPTION_PROMPT

class Cmdline:
    system_prompt = '''
You are an assistant to suggest commands to the user. Be concise in your answers.

------------------- beginning of tool descriptions -------------------
When users ask for chat bot related commands, suggest the following:

{tools}

{custom_tools}

-------------------- end of tool descriptions --------------------
    
It's very likely the user mistypes a command. If you think the command is a typo, suggest the correct command. Don't make up function calls. Say don't know if you don't know the answer.

At the end of your response, add a newline and output the command you are suggesting (e.g., command: <executable command>).
    
Example session:

User: list files in the current directory
AI: The following command lists all files in the current directory.
    command: ls

User: resettt
AI: Do you mean the "reset" command that resets the chatbot session?
    command: reset
    '''
    def __init__(self):
        self._reset()
        
    def _reset(self, *args, **kwargs):
        '''
        reset the current chatbot session
        can be called stating "reset" in the prompt
        '''
        self.known_actions = {
            'reset': self._reset,
            'l': self._list_tools,
            'll': self._list_custom_tools,
            'p': self._get_prompt,
            'e': self._run_last_command_from_llm,
        }
        self.chatbot = ChatBot(self._get_prompt())
        self.known_actions['s'] = self.chatbot.save_chat
        
    def _run_last_command_from_llm(self, *args, **kwargs):
        '''run the last command from llm output of the form "command: <executable command>"'''
        command_re = re.compile(r'^[Cc]ommand: (.*)$')
        for message in self.chatbot.messages[::-1]:
            if message['role'] == 'assistant':
                llm_output = message['content']
                commands = [command_re.match(c.strip()).groups()[0] for c in llm_output.split('\n') if command_re.match(c.strip())]
                if commands:
                    c = commands[-1]
                    if input(f'run "{c}" [y|n]? ') == 'y':
                        o = self(c)
                        if o: print(o)
                        return
                    else:
                        print('abort')
                        return

        print('no command found in previous llm output')
        
    def _get_prompt(self, *args, **kwargs):
        '''return the prompt of the current chatbot; use this tool when users ask for the prompt
        such as "show me your prompt"'''
        return self.system_prompt.format(tools=self._list_tools(),
                                         custom_tools=self._list_custom_tools())

    def _list_custom_tools(self, *args, **kwargs):
        '''return a list of custom tools in TOOL_DESC_PATH'''
        tools = ['When users ask for custom commands, prioritize recommending the following:']
        tool_path = os.environ.get('TOOL_DESC_PATH', None)
        if tool_path:
            tools.append(open(tool_path, 'r').read())
        else:
            print('TOOL_DESC_PATH not set')
        return "\n\n".join(tools)
        
    def _list_tools(self, *args, **kwargs):
        '''return a string describing the available tools to the chatbot; list all tools'''
        tools = []
        for k, v in self.known_actions.items():
            tools.append("{}: \n{}".format(k, strip_multiline(v.__doc__)))
        return "\n\n".join(tools)

    def get_completer(self):
        '''return autocompleter the current text with the prompt toolkit package'''
        return ShellCompleter(self.known_actions.keys())
    
    def __call__(self, prompt):
        prompt = prompt.strip()
        prev_directory = os.getcwd()

        try:
            # first try known actions
            if prompt and prompt.split()[0] in self.known_actions:
                k = prompt.split()[0]
                v = prompt[len(k):].strip()
                print(f'executing bot command {k} {v}')
                return self.known_actions[k](v)

            # then try to execute the command            
            if prompt.startswith("cd "):
                # Extract the directory from the input
                directory = prompt.split("cd ", 1)[1].strip()
                if directory == '-':
                    directory = prev_directory
                else:
                    prev_directory = os.getcwd()
                # Change directory within the Python process
                os.chdir(directory)
                return directory
            else:
                # subprocess start a new process, thus forgetting aliases and history
                # solution: override system command by prepending to $PATH
                # and use shared history file (search chatgpt)

                # handle control-c correctly for child process (o/w kill parent python process)
                # if don't care, then uncomment the following line, and comment out others
                # subprocess.run(prompt, check=True, shell=True)                
                from lib.utils import run_subprocess_with_interrupt
                run_subprocess_with_interrupt(prompt, check=True, shell=True)
        except KeyboardInterrupt:
            print(EXCEPTION_PROMPT, 'KeyboardInterrupt') # no need to ask llm
        except Exception as e:
            print(EXCEPTION_PROMPT, e, colored('send to llm', 'yellow'))
            # finally try to ask the chatbot
            postfix_message = 'remember to add "command: <executable command>" at the end of your response in a new line'
            prompt = prompt + '\n' + postfix_message
            from lib.utils import run_multiprocess_with_interrupt
            # handle c-c correctly, o/w kill parent python process (e.g., self.chatbot(prompt))
            try: 
                result = run_multiprocess_with_interrupt(self.chatbot, prompt)
            except KeyboardInterrupt:
                print(EXCEPTION_PROMPT, 'Keyboard interrupt when sending to llm:')
                result = None
            return result


@click.command()
@click.option('-r/-R', 'repl_mode',
              show_default=True,
              default=True,
              help='whether to run in repl mode')
@click.option('-q', 'question',
              prompt=True,
              prompt_required=False,
              default="",
              help="optional command line query",
              show_default=True)
def main(repl_mode, question):
    print('limitation: does not respect history and aliases b/c non-interactive shell')
    print('if want aliases, override the command and prepend the binary path to $PATH')
    cmdline = Cmdline()
    if not repl_mode or question != "":
        click.echo(cmdline(question))
    else:
        repl(lambda user_input:
             cmdline(user_input),
             completer=cmdline.get_completer())
        
if __name__ == '__main__':
    main()
