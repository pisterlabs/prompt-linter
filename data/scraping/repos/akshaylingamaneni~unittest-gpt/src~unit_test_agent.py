import argparse
import os

import autopep8
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

from src.file_utils import FileUtils

load_dotenv()


class UnitTestAgent:
    def generate_test_case(self, file_name, file_path):
        try:
            # Read the file contents
            self.create_tests_init_file()
            with open(file_path, 'r') as file:
                file_contents = file.read()
            source_file_name = os.path.splitext(file_name)[0]
            source_file_path = file_path
            src_index = source_file_path.find("src")
            source_file_path = source_file_path[src_index:]
            test_file_name = f"{source_file_name}_test.py"
            # Generate the test case code
            test_case_code = self.generate_test_case_code(file_contents, file_name, test_file_name,
                                                          source_file_path)
            test_case_code = self.get_cleaned_code(test_case_code)
            formatted_code = autopep8.fix_code(test_case_code)

            # Get the path to the parent directory of the "src" director
            project_directory = os.path.abspath(os.getcwd())

            # Create the "tests" directory if it doesn't exist
            tests_directory = os.path.join(project_directory, 'tests')

            # Create the tests directory if it doesn't exist
            if not os.path.exists(tests_directory):
                os.makedirs(tests_directory)

            # Write the formatted code to the test file in the tests directory
            test_file_path = os.path.join(tests_directory, test_file_name)
            with open(test_file_path, 'w') as test_file:
                test_file.write(formatted_code)

            print('Test case generated successfully.')
        except IOError as e:
            print(f'Error occurred while reading the file: {str(e)}')

    def generate_test_case_code(self, file_contents, source_file_name, test_file_name, source_file_path):
        template = """ You are a highly skilled unit test generator assistant for my codebase. You will generate pyton unit test code using the pytest library for the source code in the project. 
        A a highly skilled ai assistant you are aware of what are the best practices when writing unit tests and think through in identifying edge cases. I will provide with the sourcecode of the class for which you need to generate unittest using pytest libray.
         interested in the pytest unit test code without any accompanying source code or explanation of the code and put it in a code blocks
    
                    Best practices for writing unit tests for you knowledge
                       1: Descriptive Test Names: Use descriptive names for your test methods that clearly indicate what is being tested.
                       2. Separate Test Methods: Write separate test methods for each specific behavior or scenario you want to test.
                       3. Arrange, Act, Assert (AAA) Pattern: Structure your test methods with clear sections for arranging preconditions, acting on the code under test, and asserting the expected results.
                       4. Use Assertions: Utilize the available assertion methods (e.g., assertEqual, assertTrue, etc.) to validate expected outcomes and conditions.
                       5. Setup and Teardown: Use the setUp and tearDown methods to set up any necessary preconditions and clean up after the test.
                       6. Test Coverage: Aim for comprehensive test coverage by testing different paths, edge cases, and boundary conditions.
                       7. Isolation and Independence: Ensure each test is independent and does not rely on the state or outcome of other tests.
                       8. Documentation and Comments: Provide clear documentation or comments for your test methods to explain their purpose and any relevant considerations.
                       9. Readable and Maintainable: Write clean and readable test code, following coding standards and practices.
                       
                        Generate testcase for {source_file_name} and name output file name is {test_file_name}, also here is the path of the source file from src directory: {source_file_path} use this correctly import the source file for example src/LinkedList/valid_parentheses.py mean the you need to add the following "import from src.LinkedList.valid_parentheses import ValidParentheses"
                        below is code in the source file: 
                        {sourceCode}
    
                        Assistant:"""
        prompt = PromptTemplate(template=template, input_variables=["sourceCode", "source_file_name", "test_file_name",
                                                                    "source_file_path"])
        llm = ChatOpenAI(temperature=0, callbacks=[StreamingStdOutCallbackHandler()],
                         streaming=True)
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        test_case_code = chain.run(
            {"sourceCode": file_contents, "source_file_name": source_file_name, "test_file_name": test_file_name,
             "source_file_path": source_file_path})

        return test_case_code

    def get_cleaned_code(self, code):
        start_index = code.find("```")

        end_index = code.rfind("```")

        # Extract the code from the text
        code_block = code[start_index + 3: end_index]

        # Remove leading and trail  ing whitespace
        code = code_block.strip()
        return code

    def create_tests_init_file(self):
        project_directory = os.path.abspath(os.getcwd())

        # Create the "tests" directory if it doesn't exist
        tests_directory = os.path.join(project_directory, 'tests')

        if not os.path.exists(tests_directory):  # Check if the tests directory doesn't exist
            os.makedirs(tests_directory)  # Create the tests directory if it doesn't exist

        init_file = os.path.join(tests_directory, '__init__.py')  # Path to the __init__.py file

        if not os.path.isfile(init_file):  # Check if the __init__.py file doesn't exist
            with open(init_file, 'w'):  # Create an empty __init__.py file
                pass

    def generate_testcases(self, path=None, file=None):
        if path is None and file is None:
            raise ValueError("Either 'path' or 'file' parameter must be provided.")

        if path is not None and file is not None:
            raise ValueError("Only one of 'path' or 'file' parameters should be provided, not both.")

        file_utils = FileUtils()
        py_file_dict = None
        if path is not None:
            directory = file_utils.find_src_directory(path)
            py_file_dict = file_utils.collect_py_files(directory)
        if file is not None:
            if py_file_dict is None:
                directory = file_utils.find_src_directory("src")
                py_file_dict = file_utils.collect_py_files(directory)
            if file.endswith('.py'):
                file_name = file
                file_path = py_file_dict.get(file_name[:-3])
                if file_path is not None:
                    self.generate_test_case(file_name, file_path)
                else:
                    print(f"File '{file_name}' not found in the directory.")
            else:
                print(f"Invalid file extension. Expected '.py' file, got '{file}'.")
        else:
            for entry, value in py_file_dict.items():
                file_name = f"{entry}.py"
                file_path = value
                self.generate_test_case(file_name, file_path)


if __name__ == "__main__":
    agent = UnitTestAgent()

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the desired command-line options
    parser.add_argument('-f', '--file', help='Specify the file name')
    parser.add_argument('-dir', '--directory', help='Specify the directory')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check the provided options
    if args.file:
        # Call the method with the file parameter
        agent.generate_testcases(file=args.file)
    elif args.directory:
        # Call the method with the directory parameter
        agent.generate_testcases(path=args.directory)
    else:
        # No valid options provided, raise an error or handle it accordingly
        raise ValueError("Please provide either a file or directory option.")
