import os
import re
import sys
import random
from typing import Optional
from pathlib import Path
from langchain_g4f import G4FLLM
from g4f import ChatCompletion, models, Provider
import autopep8
import coverage

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from test_g4f_providers import ProviderTester

"""
TODOS:
- Add support for improving all files in a given directory.
- Add the version to the new file name suffix.
- Add a method to check for code syntax errors using the compile function.
- Add a method to format code using autopep8 or black.
- Add a method to generate a code coverage report.
- Add self improve mode. When path input is empty process the script's own path.
"""
class CodeImprover:
    """
    A pragmatic, procedural, and organized code analyzing, improving and upgrading agent.
    """
    version: Optional[float] = 2.3
    script_path: Optional[str] = None
    log_file: Optional[str] = "changes.txt"

    def __init__(self):
        self.python_mark: Optional[str] = r"```(python|py|)\n(?P<code>[\s\S]+?)\n```"

    @staticmethod
    def read_code(text: str, code_mark: str) -> str:
        match = re.search(code_mark, text)
        if match:
            return match.group("code")

    def save_changes(self, text: str) -> None:
        with open(self.log_file, "a") as file:
            file.write(text)
        print(f"Changes saved to log {self.log_file}")

    def check_syntax_errors(self, text: str) -> None:
        try:
            compile(text, "<string>", "exec")
        except SyntaxError as e:
            print(f"Syntax Error: {e}")

    def format_code(self, code: str) -> str:
        return autopep8.fix_code(code)

    def generate_coverage_report(self, code: str) -> None:
        with open("temp.py", "w") as file:
            file.write(code)

        cov = coverage.Coverage()
        cov.start()
        os.system("python temp.py")
        cov.stop()
        cov.save()
        cov.report()

        os.remove("temp.py")
        os.remove(".coverage")

    def improve_code(self) -> None:
        """ Improves the code in a given file. """
        if self.script_path and os.path.exists(self.script_path):
            path = self.script_path
        else:
            path = input("Enter the path of the file you want to improve:(enter to self-improve) ")
            if path == "":
                path=str(Path(__file__))

        try:
            with open(path, "r") as file:
                code = file.read()

            prompt = f"""
            You are a pragmatic, procedural, and organized code analyzing and improving agent. 
            You can ingest, analyze, improve and upgrade scripts.
            Rules:
            - Don't make a new comment block at the end of the script if one exists, just add or subtract info as needed.
            - Don't change methods, classes or variables that have '#protected' above the definition.
            - Don't remove any functionality only add functionality.
            - Don't remove the imports at the top of the script.
            - Don't add type hints to kwargs.
            - Don't remove license comments.
            - The 'Description:' in the bottom 'comment block' is where you describe the script
            - The 'Usage:' description in the bottom 'comment block' is where you explain the usage.
            - The 'Use cases:' in the bottom 'comment block' is where you list predicted use cases.
            - The 'Proposed extra features:' in the bottom 'comment block' is where you list proposed new features.
            - The 'TODOS:' in the top 'comment block' is where the user will list the todo's.
            - Create or Update the 'version' variabe by adding 0.1 to it.
            - List the changes you made starting with 'I made the following changes:'.
            - The 'comment block' at the top of the script is always formatted like this:
                f'''
                TODOS:
                    <here the user places todos he wants implemented>
                '''
            - The 'comment block' at the bottom of the script is always formatted like this:
                f'''
                Description:
                    <here the assistant describes script working>
                Usage:
                    <here the assistant describes script usage>
                Predicted use cases:
                    <here the assistant describes use cases>
                Proposed features:
                    <here the assistant proposes features>
                '''
            """
            improve_task=f"""
            Tasks:
            - 1 Refactor the code into a callable class if needed.
            - 2 Add a __name__ == "__main__" if needed.
            - 3 Implement the items the user listed under 'TODOS:' in the top 'comment block'.
            - 4 Add or change variables, methods, enums, classes and logic to improve and enhance the script.
            - 5 Add type hints.
            - 6 Update the bottom 'comment block' with relevant information.
            Now improve the code in this file:

            ```py
            {code}
            ```
            """

            response = []
            provider = self.pick_random_provider()
            model = models.gpt_35_turbo #models.gpt_35_turbo.name if provider.supports_gpt_35_turbo else models.default.name
            #print(provider)
            #print(model)
            for chunk in ChatCompletion.create(
                
                model = model,
                #provider = Provider.retry_provider.RetryProvider,
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": improve_task}],
                timeout=6000,
                stream=False,
            ):
                response.append(chunk)
                print(chunk, end="", flush=True)

            response = "".join(response)

            code = self.read_code(response, self.python_mark)
            if code:
                # Check for syntax errors
                self.check_syntax_errors(code)

                # Format the code
                code = self.format_code(code)

                # Generate a code coverage report
                self.generate_coverage_report(code)

                new_file_path = str(Path(path).with_name(f"{Path(path).stem}_generated_improvement{Path(path).suffix}"))
                with open(new_file_path, "w") as file:
                    file.write(code)
                print(f"Improved code saved to {new_file_path}")

            # split of the changes and save them
            changes = response.split("I made the following changes:")
            changes = f"I made the following changes in version {self.version}:\n{changes.pop()}"
            self.save_changes(changes)

        except FileNotFoundError:
            print("Invalid file path.")

    def pick_random_provider(self):
        if os.path.exists("live_providers.txt"):
            with open("live_providers.txt", "r") as file:
                providers = file.read()
        else:
            # Create an instance of ProviderTester with all the providers
            tester = ProviderTester().main()
        pick = providers[random.randint(0, len(providers)-1)]
        if not "error" in pick and pick != "Wewordle":
            return pick
        else:
            self.pick_random_provider()

if __name__ == "__main__":
    CodeImprover().improve_code()

f'''
Description:
A pragmatic, procedural, and organized code analyzing, improving and upgrading agent.

Usage:
- Run the script.
- Enter the path of the file you want to improve when prompted.
- The script will use the language model to generate an improved version of the code.
- The improved code will be saved in a new file with a suffix "_generated_improvement" in the file name.

Predicted use cases:
- Improving code by analyzing and providing automated suggestions.
- Learning and understanding the improvements made by the language model.

Proposed extra features:
- Add support for specifying the language model to use.
- Add support for customizing the code improvement prompt.
- Add a method to run unit tests for
'''