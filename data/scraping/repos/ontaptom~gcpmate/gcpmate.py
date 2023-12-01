"""
GCPMate - Google Cloud Platform assistant.

This module provides functions and classes to assist with managing 
Google Cloud Platform using natural language queries and OpenAI's GPT-3 language model.
"""

import os
import re
import sys
import subprocess
import argparse
import shlex
from time import sleep
from prettytable import PrettyTable
import openai


class GCPMate:
    """
    GCPMate is an OpenAI-powered assistant for managing Google Cloud Platform resources.
    """
    
    def __init__(self, openai_model="text-davinci-003", skip_info=False):
        """
        Initializes a new instance of the GCPMate class with the specified OpenAI model.

        Args:
            openai_model (str): The name of the OpenAI model to use for generating gcloud commands.
            skip_info (bool): Flag indicating whether or not to skip printing runtime info.
        """

        try:
            self.current_user = subprocess.run(
                ['gcloud', 'auth', 'list', '--filter=status:ACTIVE',
                    '--format=value(account)'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            ).stdout.decode('utf-8').strip()
            self.current_project = subprocess.run(
                ['gcloud', 'config', 'get-value', 'project'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            ).stdout.decode('utf-8').strip()
            self.default_region = subprocess.run(
                ['gcloud', 'config', 'get-value', 'compute/region'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            ).stdout.decode('utf-8').strip()
            self.default_region = "(unset)" if self.default_region == "" else self.default_region
            self.default_zone = subprocess.run(
                ['gcloud', 'config', 'get-value', 'compute/zone'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            ).stdout.decode('utf-8').strip()
            self.default_zone = "(unset)" if self.default_zone == "" else self.default_zone
            self.gcloud_available = True
        except FileNotFoundError:
            self.current_user = "gcloud not found"
            self.current_project = "gcloud not found"
            self.default_region = "gcloud not found"
            self.default_zone = "gcloud not found"
            self.gcloud_available = False
        self.openai_model = openai_model
        self.skip_info = skip_info
        self.commands = []

    def blue_text(self, text):
        """
        Returns the specified text in blue for console output.
        """

        return f"\033[94m{text}\033[0m"
    
    def animate(self, text):
        """
        Animates the specified text in the console.
        """

        for char in text:
            sleep(0.05)
            print(char, end='', flush=True)
        print()

    def get_yes_no(self):
        """
        Asks the user to confirm whether or not to execute a set of gcloud commands.
        """

        while True:
            if not self.skip_info:
                print(f"\n{self.blue_text('Fair warning')}: Execute the command(s) only if "
                    "fully understand the consequences. \n\t gcloud may prompt for yes/no "
                    "confirmation. If so, execution process will respond with yes.\n")
            answer = input(
                f"Would you like to execute the following {self.blue_text(len(self.commands))} "
                 "command(s)? [y/N] ").strip().lower()
            if answer in {"y", "yes"}:
                return True
            elif answer in {"", "n", "no"}:
                return False
            else:
                print("Invalid input, please try again.")

    def call_openai_api(self, query):
        """
        Calls the OpenAI API to generate gcloud commands based on the specified query. 
        Since returned output is a multiple-line string, it is split into a list of 
        commands and stored in the self.commands variable.
        """

        try:
            response = openai.Completion.create(
                model=self.openai_model,
                prompt=query,
                temperature=0,
                max_tokens=350,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
        except Exception as api_error:
            print("Error with OpenAI API request: ", api_error)
            sys.exit(1)

        return response['choices'][0]['text']

    def generate_commands(self, api_response):
        """
        Assuming api_response contains list of gcloud commands. This method removes unnecessary 
        characters from the OpenAI API response, splits the response into a list of 
        commands, and stores the list in the self.commands variable.
        """

        # remove \<new-line> in case if OpenAI returns gcloud in multiple lines
        singleline_commands = api_response.replace(
            '\\\n', '')

        # replace multiple spaces with single-space, if any found in the reply:
        singleline_commands = re.sub(' +', ' ', singleline_commands)

        # Split gcloud commands separated by '&&' to separate lines, but ignore '&&'
        # within parameter values. For example:
        # [...] --metadata startup-script='sudo apt-get update && sudo apt-get install -y nginx'
        singleline_commands = singleline_commands.replace("&& gcloud", "\n gcloud")

        # split multiple commands to a list of commands and return the list
        return [x.strip() for x in re.findall(
            r'(?:gcloud|gsutil)\b.*?(?:\n|$)', singleline_commands)]

    def print_runtime_info(self):
        """
        Prints runtime info about the current gcloud configuration.
        """

        table = PrettyTable()
        table.field_names = ["Configuration", "Value"]
        table.add_row(["Active gcloud account",
                      self.blue_text(self.current_user)])
        table.add_row(
            ["Default project", self.blue_text(self.current_project)])
        table.add_row(["Default region", self.blue_text(self.default_region)])
        table.add_row(["Default zone", self.blue_text(self.default_zone)])
        table.add_row(["OpenAI model", self.blue_text(self.openai_model)])
        table.align = "l"
        print(table)

    def execute_commands(self):
        """
        Executes the list of gcloud commands stored in the self.commands variable. If a command 
        contains a prompt, it is executed with a default response of "y".If a command contains 
        a pipe (|), it is split into subcommands and executed as a pipeline. However, if command 
        contains a pipe, and it contains a prompt, the command will not execute properly. 
        This is a known issue and will be addressed in a future release.
        """

        for command in self.commands:
            print(f"---\nExecuting: {self.blue_text(self.multiline_output(command))}")
            if "|" in command:
                subcommands = command.split("|")
                p = subprocess.Popen(shlex.split(
                    subcommands[0]), stdout=subprocess.PIPE)
                for c in subcommands[1:]:
                    p1 = subprocess.Popen(shlex.split(
                        c), stdout=subprocess.PIPE, stdin=p.stdout)
                    p.stdout.close()
                    p = p1
                try:
                    output = p.communicate()[0].decode('utf-8')
                    print(f"---\nResult:\n\n{self.blue_text(output)}")
                except subprocess.CalledProcessError as process_error:
                    print(f"---\nError: {process_error.stderr.decode('utf-8')}")
            else:
                try:
                    p1 = subprocess.run(shlex.split(command), input='y'.encode(
                    ), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    print(
                        f"---\nResult:\n\n{self.blue_text(p1.stdout.decode('utf-8'))}\n"
                        f"{self.blue_text(p1.stderr.decode('utf-8'))}")
                except subprocess.CalledProcessError as process_error:
                    print(f"---\nError: {process_error.stderr.decode('utf-8')}")

    def multiline_output(self, command, sep=' \\ \n\t'):
        """
        Check if command is 100 characters or more, if so, it adds ' \\ \n\t' or other separator
        at the nearest space to the n * 100th character. This is to print command
        in multiple lines in the terminal.
        """

        if len(command) < 100:
            return command
        else:
            lines = []
            while len(command) > 100:
                lines.append(command[:command[:100].rfind(' ')] + sep)
                command = command[command[:100].rfind(' ')+1:]
            lines.append(command) # add the last line
            return ''.join(lines)

    def explain(self,query):
        """
        Explain the query to the user
        """
        response = self.call_openai_api(query)
        response = response.lstrip() + "\n" # response sometimes contains unnecessary leading spaces
        self.animate(self.blue_text(self.multiline_output(response, sep="\n")))

    def run(self, query):
        """
        Main method to run GCPMate with the specified query.

        Args:
            query (str): The query to be passed to the OpenAI API.
        """

        if not self.skip_info:
            self.print_runtime_info()

        # call OpenAI API
        api_response = self.call_openai_api(query)

        # generate list of commands from the API response
        self.commands = self.generate_commands(api_response)


        if len(self.commands) == 0:
            print("I'm sorry. Your question did not return any potential solution.\n"
                  "You can try rephrasing your question or use a different model by running the "
                  "command with '-m <model_name>' parameter. For more info run 'gcpmate -h'.")
            # finish script at this point
            return

        print(
            f"The proposed solution consist of {len(self.commands)} command(s):")
        i = 0
        for command in self.commands:
            i += 1
            self.animate(f'\t[{i}] {self.blue_text(self.multiline_output(command))}')

        if self.gcloud_available:
            doit = self.get_yes_no()
        else:
            doit = False
            print("gcloud is not found, bye. ")
            return
        if not doit:
            # placeholder for exit message
            return
        else:
            self.execute_commands()

def main():
    """ Main function to run GCPMate."""
    
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        print("GCPMate uses OpenAI API to assist user with Google Cloud mgmt. To use this tool "
              "please set OPENAI_API_KEY environment variable to your OpenAI API key.\n"
              "You can get your API key from https://platform.openai.com/account/api-keys. "
              "To set the environment variable, run the following command:\n\n"
              "export OPENAI_API_KEY=<your-api-key>\n")
        sys.exit(1)
    openai.api_key = openai_api_key

    parser = argparse.ArgumentParser(description='GCPMate - Google Cloud Platform assistant.\n'
                                     'Describe in query what you wish to achieve, and gcpmate '
                                     '(with a little help from OpenAI) will try to come up with a solution.\n'
                                     'If you like proposed outcome, gcpmate can also '
                                     'handle execution!', add_help=True,
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog='Example usage:\n\ngcpmate "create new project called '
                                     'my-superb-new-project"')
    parser.add_argument(
        'query', type=str, help='Query explaining what you wish to achieve in GCP')
    parser.add_argument('-m', '--model', type=str, help='OpenAI model to use for completion. Default: text-davinci-003. '
                        'Also available: code-davinci-002')
    parser.add_argument('-s', '--skip-info', action='store_true',
                        help='Skip printing "Fair warning" message and runtime info (gcloud account, project, region, zone, OpenAI model)')
    parser.add_argument('-e', '--explain', action='store_true',
                        help='Returns explanation to given query, which can be command, error message, etc.')
    args = parser.parse_args()

    model = args.model if args.model else "text-davinci-003"

    gcpmate = GCPMate(openai_model=model, skip_info=args.skip_info)
    if args.explain:
        full_query = f"""
            Context: Explain the following.
            Prompt: {args.query}
            Explaination:
            """
        gcpmate.explain(full_query)
    else:
        full_query = f"""
                Context: Provide only gcloud command as output.
                Prompt: {args.query}
                Command: gcloud
                """

        gcpmate.run(full_query)

if __name__ == '__main__':
    main()
