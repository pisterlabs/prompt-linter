import click
from langchain import PromptTemplate, LLMMode, LLMChain
from langchain.llms import CoCode
from typing import List, Optional


class Architect:
    @PromptTemplate
    def update_diagram(diagram, code):
        return f"""
        Given this diagram:
        
        {diagram}

        And this Python code:

        {code}

        Update the diagram to match the code implementation. Use consistent naming, show return types on arrows, and indicate the flow of data.
        """


class Coder:
    def generate_code(diagram: str) -> str:
        """Generate initial code structure"""
        return f"""
        #!/usr/bin/env python3
        # Imports for the major classes/components
        import typing
        
        # Global config

        # Constants

        # Types

        # Classes & methods

        # Functions

        # Main function

        # Main runner code
        if __name__ == "__main__":
            main()
        """

    @PromptTemplate
    def update_code(diagram, code):
        return f"""
        Given this updated diagram:
        
        {diagram}
        
        And current Python code:

        {code}

        Modify the Python code to match the diagram. Ensure the functions, returns, and parameters align with what is shown in the diagram.
        """


class Tester:
    @PromptTemplate
    def validate(diagram, code):
        return f"""
        Given this diagram:
        
        {diagram}
        
        And Python code:

        {code}

        Ensure the functions, returns, and parameters align with what is shown in the diagram.
        """


@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.argument("diagram_file", type=click.File("r"))
@click.command()
def main():
    code = Coder.generate_code(diagram)

    chain = LLMChain(LLMMode.Dialog, lm=CoCode)

    iterations = 10  # TODO: Reach a fixed point of code as noted by Tester.validate
    for i in range(iterations):
        diagram = chain.prompts(Architect.update_diagram(diagram, code))
        code = chain.prompts(Coder.update_code(diagram, code))

    valid = chain.prompts(Tester.validate(diagram, code))


if __name__ == "__main__":
    main()
