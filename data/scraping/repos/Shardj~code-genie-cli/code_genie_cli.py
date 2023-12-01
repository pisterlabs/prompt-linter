import os, sys, re, clipboard, subprocess
from typing import Dict, Optional, Any, List, Tuple
from definitions import KEY_PATH, DEBUG
from colorama import Fore, Style
# For some reason Fore.RESET is actually secretly Style.RESET_ALL and this is undocumented behoaviour.
# So we need to manually set Fore.RESET to the correct value which will only reset the foreground colour and not touch the style.
Fore.RESET = "\033[39m"
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import ANSI 
from code_genie_cli.openai_api_caller import OpenaiApiCaller
from code_genie_cli.code_executor import CodeExecutor
from code_genie_cli.system_content import SystemContent
from code_genie_cli.spinner import Spinner

# Create a custom key binding to allow multiline input
bindings = KeyBindings()

@bindings.add('c-v')
def _(event):
  event.current_buffer.insert_text(clipboard.paste())

@bindings.add('escape', 'c-m', eager=True)
def _(event):
  event.current_buffer.newline()

@bindings.add('c-m', eager=True)
def _(event):
  buf = event.current_buffer
  if buf.complete_state:
    buf.complete_next()
  else:
    event.app.exit(result=buf.text)


# Main class, used by calling run() on an instance of this class.
# This class mainly handles the user input and the response from OpenAI
class CodeGenieCLI:
  def __init__(self) -> None:
    # The OpenaiApiCaller instance handles messy things like reading the API key and keeping track of the chat history
    self.openai_api_caller = OpenaiApiCaller()
    self.code_executor = CodeExecutor()
    self.spinner = Spinner()

  def run(self) -> None:
      try:
        self.__clear_terminal()
        print(f"{Style.BRIGHT}{Fore.GREEN}Welcome to {Fore.BLUE}code-genie-cli{Fore.GREEN}!{Fore.RESET}")
        # Our first prompt will be the system message, this gets genie to introduce themselves to the user as well as allowing us to calculate how many tokens it is
        self.__chat_ask_and_response_handling(SystemContent().generate(), "system")
        first_promt_injection = " (alt + enter for new line)"
        # At it's most basic, we simply loop over the user input and the genie response. Forever.
        while True:
          # TODO: seems to be some weird behavior where the cursor doesn't move to the next character on first key press. So the 2nd character then overwrites it.
          # However this is only a visual thing and when you hit enter the characters all re-appear
          user_message = prompt(ANSI(f"\n{Style.BRIGHT}{Fore.GREEN}User{first_promt_injection}:{Fore.RESET} "), key_bindings=bindings)
          first_promt_injection = ""
          self.__chat_ask_and_response_handling(user_message)
      except KeyboardInterrupt:
        print(f"{Fore.YELLOW}\n\nExiting the script gracefully.{Style.RESET_ALL}")
        sys.exit(0)

  def __clear_terminal(self):
    if os.name == 'posix':  # for Linux and macOS
        _ = subprocess.call('clear')
    elif os.name == 'nt':  # for Windows
        _ = subprocess.call('cls', shell=True)

  def __chat_ask_and_response_handling(self, user_message: Optional[str] = None, role: str = "user") -> None:
    print(Fore.BLUE, end="")
    self.spinner.continue_spinner()
    response = self.openai_api_caller.chat(user_message, role)
    self.spinner.halt_spinner()
    print(Fore.RESET, end="")

    # Find all code blocks within triple backticks
    code_blocks = re.findall(r"(```\S*\n)([\s\S]*?)(```)", response)

    # Replace code blocks with magenta-colored content
    colored_response = response
    for opening, content, closing in code_blocks:
        colored_content = f"{Fore.MAGENTA}{content}{Fore.RESET}"
        colored_code_block = f"{opening}{colored_content}{closing}"
        colored_response = colored_response.replace(f"{opening}{content}{closing}", colored_code_block)

    print(Fore.BLUE + "\nGenie:\n" + Fore.RESET + colored_response)

    # We ignore any words on the same line as the opening backticks, as they are likely to be a language specifier
    # So for example ```python is treated the same as ```
    code_blocks = re.findall(r"```\S*([\s\S]*?)```", response)
    if code_blocks:
        # Merge all code blocks into a single block, separated by a newline character
        merged_code_blocks = "\n".join(code.strip() for code in code_blocks)

        action = input(f"{Fore.CYAN}\nExecute the provided code? (y/n) {Fore.RESET}").lower()
        if action == "y":
            self.__execute_code_with_chat_output(merged_code_blocks)

  def __execute_code_with_chat_output(self, code: str) -> None:
    print(Fore.CYAN + f"\nExecution output: {Fore.RESET}")
    success, output = self.code_executor.execute_code(code)
    if output.strip():
      action = input(f"{Fore.GREEN}\nWould you like to give the {'output' if success else 'error'} back to {Fore.BLUE}Genie{Fore.GREEN}? (y/n) {Fore.RESET}").lower()
      if action == "y":
        if success:
          self.__chat_ask_and_response_handling(f"The code execution outputed: \n{output}")
        else:
          bonus = ""
          if 'ModuleNotFound' in output:
            bonus = "Please add a try except block to the broken import, on except you should install the package and import again. If you have already tried this then stop using that package."
          else:
            bonus = "Please fix your code and try again. Provide a single python script to solve the users request."
          self.__chat_ask_and_response_handling(f"An error occoured. {bonus} Here's the output: \n{output}")
    else:
      print(f"No output from code execution.")
