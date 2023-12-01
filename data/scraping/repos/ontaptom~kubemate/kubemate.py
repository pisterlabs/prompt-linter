"""
kubemate - Kubernetes assistant.

This module provides functions and classes to assist with managing 
Kubernetes using natural language queries and OpenAI's GPT-3 language model.
"""

import os
import re
import sys
import secrets
import argparse
from time import sleep
import openai


class KubeMate:
    """
    kubemate is an OpenAI-powered assistant for managing Kubernetes resources.
    """

    def __init__(self, openai_model="text-davinci-003"):

        self.openai_model = openai_model


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
            sleep(0.03)
            print(char, end='', flush=True)
        print()

    def propose_filename(self):
        """
        Proposes a filename for the YAML output to be saved at..
        """

        random_filename = secrets.token_hex(3)

        return f"/tmp/kubemate_{random_filename}.yaml"

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

        # print("\n\n debug \n\n")
        # print(response['choices'][0])
        # print("\n\n debug \n\n")

        return response['choices'][0]['text']

    def adjust_yaml(self, yaml):
        """
        Adjusts the YAML returned by the OpenAI API to be more readable.
        """

        # find if the whole output has unnecessary leading spaces
        leading_spaces = re.search(r'\n\s+(?=apiVersion)', yaml)
        if leading_spaces:
            yaml = yaml.replace(leading_spaces.group(0), '\n')

        # add new line at the end of yaml
        if not yaml.endswith('\n'):
            yaml += '\n'
        
        # remove new line if yaml starts with \n
        if yaml.startswith('\n'):
            yaml = yaml[1:]

        return yaml

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
        Main method to run kubemate with the specified query.

        Args:
            query (str): The query to be passed to the OpenAI API.
        """

        # call OpenAI API
        api_response = self.call_openai_api(query)

        # adjust YAML returned by the API
        api_response = self.adjust_yaml(api_response)
        # generate list of commands from the API response
        print("Proposed solution:")
        #print(api_response)
        self.animate(self.blue_text(api_response))

        answer = input("Would you like to save this solution to a file? [y/N]: ").strip().lower()
        if answer in {"y", "yes"}:
            proposed_filename = self.propose_filename()
            filename = input(f"Filename [{self.blue_text(proposed_filename)}]: ").strip()
            if not filename:
                filename = proposed_filename
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(api_response)
            except Exception as file_error:
                print(f"Error writing to file: {file_error}")
                sys.exit(1)

            print(f"Solution saved to {self.blue_text(filename)}")



def main():
    """ Main function to run kubemate."""

    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        print("kubemate uses OpenAI API to assist user with K8s management. To use this tool "
              "please set OPENAI_API_KEY environment variable to your OpenAI API key.\n"
              "You can get your API key from https://platform.openai.com/account/api-keys. "
              "To set the environment variable, run the following command:\n\n"
              "export OPENAI_API_KEY=<your-api-key>\n")
        sys.exit(1)
    openai.api_key = openai_api_key

    parser = argparse.ArgumentParser(description='kubemate - Kubernetes assistant.\n'
                                     'Describe in query what you wish to achieve, and kubemate '
                                     '(with a little help from OpenAI) will try to come up with a solution.\n'
                                     'In form of YAML file. kubemate can also '
                                     'handle execution!', add_help=True,
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog='Example usage:\n\nkubemate "create an nginx deployment called '
                                     'nginx with 3 replicas and expose it as a service on port 80"')
    parser.add_argument(
        'query', type=str, help='Query explaining what you wish to deploy in K8s')
    parser.add_argument('-e', '--explain', action='store_true',
                    help='Returns explanation to given query, which can be command, error message, etc.')
    
    args = parser.parse_args()

    model = "text-davinci-003"
    # model = "code-davinci-002"

    kubemate = KubeMate(openai_model=model)
    if args.explain:
        full_query = f"""
            Context: Explain the following.
            Prompt: {args.query}
            Explaination:
            """
        kubemate.explain(full_query)
    else:
        full_query = f"""
                Context: Return only YAML for kubernetes objects:
                Prompt: {args.query}
                """

        kubemate.run(full_query)

if __name__ == '__main__':
    main()
