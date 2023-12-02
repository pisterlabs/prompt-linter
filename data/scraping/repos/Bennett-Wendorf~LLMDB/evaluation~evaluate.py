import os
import ast
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from typing import List

from llms.NonChatOpenAILangchainAgent import NonChatOpenAILangchainAgent
from llms.GPT3LangchainAgent import GPT3LangchainAgent
from llms.GPT4LangchainAgent import GPT4LangchainAgent
from llms.GPT4TurboLangchainAgent import GPT4TurboLangchainAgent

from sql import SQL
from sql import Question

base_class_name = "BaseLLM"

directory_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./llms")

evaluator = ChatOpenAI(temperature=0)

prompt_template = PromptTemplate.from_template("Answer only yes or now, with no explanation or punctuation.\nDo the following two statements convey the same meaning?\n{human_answer}\n{llm_answer}")

chain = prompt_template | evaluator

sql_agent = SQL()

questions = sql_agent.get_questions()

def test(class_name):
    # Get a reference to the class by its name
    class_ref = globals().get(class_name)

    if class_ref is not None:
        # Create an instance of the class
        instance = class_ref()

        print(f"Testing {class_name}...")

        labeled_questions = [question for question in questions if question.human_answer is not None and question.human_answer != ""]

        for i, question in enumerate(labeled_questions):
            print(f"Question {i + 1}: {question.question_text}")

            try:
                llm_answer = instance.inference(question.question_text)
            except Exception as e:
                print(f"An error occurred: {e}")
                sql_agent.add_evaluation(question.id, False, class_name, question.human_answer, exception=str(e))
                continue

            response = chain.invoke({ "human_answer": question.human_answer, "llm_answer": llm_answer }).content

            print(f"Are the answers similar: {response}")
            print()

            if response:
                success = True if "yes" in response.lower() else False
                sql_agent.add_evaluation(question.id, success, class_name, question.human_answer, lLM_answer=llm_answer)
    else:
        raise Exception(f"Class {class_name} not found")

def is_subclass(class_node, base_class_name):
    return any(
        base.id == base_class_name
        for base in class_node.bases
        if isinstance(base, ast.Name)
    )

def process_selection(selection: str, python_files: List[str]):
    if selection.lower() == "all":
        return python_files

    return [file_name for i, file_name in enumerate(python_files) if str(i+1) in selection.split(" ")] 

if __name__== "__main__":
    python_files = [file for file in os.listdir(directory_path) if file.endswith(".py") and file != "BaseLLM.py"]

    print("The following LLMs are available for testing:")
    for i, file in enumerate(python_files):
        print(f"({i+1}) {file}")
    selection = input("Classes to test (eg: \"1 2 3\", \"all\", \"All\"): ")

    selected_files = process_selection(selection, python_files)

    print()
    print("Testing the following LLMs:")
    for file in selected_files:
        print(file)
    print()

    for file in selected_files:
        file_path = os.path.join(directory_path, file)

        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name

                    if is_subclass(node, base_class_name):
                        test(class_name)