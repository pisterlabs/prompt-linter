import os
import argparse
import random
from dotenv import load_dotenv

from langchain import LLMChain
from langchain.document_loaders import GitLoader, DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from functools import reduce

from langchain.document_loaders import UnstructuredHTMLLoader
import requests
import sys

load_dotenv()


TEST_SCENARIO_COUNT = os.getenv("TEST_SCENARIO_COUNT")
GIT_REPOSITORY = os.getenv("GIT_REPOSITORY")

OPENAI_MODELS = {
    "DAVINCI": "text-davinci-003",
    "GPT-3": "gpt-3.5-turbo",
    "GPT-4": "gpt-4"
}

# MANUALLY CHANGE THIS
WEB_NAME = "python_tutorial"
WEB_LINK = "https://www.w3schools.com/python/python_file_write.asp"

def generate_file_summary_yaml():
    repo_name = WEB_NAME
    url = requests.get(WEB_LINK)
    htmltext = url.text

    gen_html_path = f"./generated/{repo_name}/file_summary/{repo_name}.html"
    os.makedirs(os.path.dirname(gen_html_path), exist_ok=True)

    f = open(gen_html_path, "w")
    f.write(htmltext)
    f.close()

    loader = UnstructuredHTMLLoader(gen_html_path)

    data = loader.load()

    query = """
    You are a professional Software Engineer who has scarce knowledge about E2E testing and file summarization.
    Create a definition file that summarizes and explains about file in yaml format.
    In the generated yaml format text, it should contain important information about the file.
    Start with a description of entire file, and then write sections of the file and their descriptions
    
    [File : {file_path}]
    {page_content}

    File Summary in yaml Format:
    """
    template = PromptTemplate(template=query, input_variables=[
                              'file_path', 'page_content'])

    openai = OpenAI(model_name="gpt-3.5-turbo")
    chain = LLMChain(prompt=template, llm=openai)

    i = 0
    for file in data:
        page_content = file.page_content
        file_path = repo_name + "_A_" + str(i)
        print(file_path)

        gen_file_path = f"./generated/{repo_name}/file_summary/{file_path}.yaml"

        os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)
        with open(gen_file_path, "w") as f:
            f.write(
                chain.run({'file_path': file_path, 'page_content': page_content}))
            f.close()
        i += 1


def fetch_file_elements_yaml():
    repo_name = WEB_NAME
    url = requests.get(WEB_LINK)
    htmltext = url.text

    gen_html_path = f"./generated/{repo_name}/{repo_name}.html"
    os.makedirs(os.path.dirname(gen_html_path), exist_ok=True)

    f = open(gen_html_path, "w")
    f.write(htmltext)
    f.close()

    loader = UnstructuredHTMLLoader(gen_html_path)
    #loader = UnstructuredHTMLLoader(gen_html_path, mode="elements")

    data = loader.load()

    query = """
    You are a professional Software Engineer who has scarce knowledge about E2E testing and file summarization.
    Minimize the file into just its interactve elements and briefly describe their function.
    In the generated yaml file, group the elements with similiar functions together in one section and include the section number and description.
    If section have more than 5 elements, keep only 5 random elements.

    Each section's format have to be like below, until triple dot (...):
    
    1. Section: ["section name"]
       Elements: 
        - [element1 : "element1 description"]
        - [element2 : "element2 description"]
    ...
    
    [File : {file_path}]
    {page_content}

    File Summary in yaml Format:
    """
    template = PromptTemplate(template=query, input_variables=[
                              'file_path', 'page_content'])

    openai = OpenAI(model_name="gpt-3.5-turbo")
    chain = LLMChain(prompt=template, llm=openai)

    i = 0
    for file in data:
        page_content = file.page_content
        file_path = repo_name + "_B_" + str(i)
        print(file_path)

        gen_file_path = f"./generated/{repo_name}/file_summary/{file_path}.yaml"

        os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)
        with open(gen_file_path, "w") as f:
            f.write(
                chain.run({'file_path': file_path, 'page_content': page_content}))
            f.close()
        i += 1

def fetch_file_sects_yaml():
    repo_name = WEB_NAME
    url = requests.get(WEB_LINK)
    htmltext = url.text

    f = open("temporaryfile.html", "w")
    f.write(htmltext)
    f.close()

    loader = UnstructuredHTMLLoader("temporaryfile.html")
    #loader = UnstructuredHTMLLoader("temporaryfile.html", mode="elements")

    data = loader.load()

    file_summary_path = f"./generated/{repo_name}/file_summary/{repo_name}_A_0.yaml"

    with open(file_summary_path, "r") as f:
        context = f.read()

    query = """
    You are a professional Software Engineer who has scarce knowledge about E2E testing and file summarization.
    Collect all interactive elements in FILE1 and group them based on the sections in FILE2.
    The elements of a section should be listed.
    
    FILE1:
    {page_content}

    FILE2:
    {context}

    File Summary in yaml Format:
    """
    template = PromptTemplate(template=query, input_variables=[
                              'context', 'page_content'])

    openai = OpenAI(model_name="gpt-3.5-turbo")
    chain = LLMChain(prompt=template, llm=openai)

    i = 0
    for file in data:
        page_content = file.page_content
        file_path = repo_name + "_B_" + str(i)
        print(file_path)

        gen_file_path = f"./generated/{repo_name}/file_summary/{file_path}.yaml"

        os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)
        with open(gen_file_path, "w") as f:
            f.write(
                chain.run({'context': context, 'page_content': page_content}))
            f.close()
        i += 1

# This function is almost identical as the one in yaml_gen.py
# However, the model used is "gpt-3.5-turbo"
def create_test_scenario():
    repo_name = WEB_NAME
    file_summary_path = f"./generated/{repo_name}/file_summary"

    if os.path.exists(file_summary_path):
        loader = DirectoryLoader(file_summary_path, loader_cls=TextLoader)
    else:
        raise Exception(
            'file_summary does not exist. Please run summary-gen to generate file_summary.')

    prompt_template = """You are a Software Engineer who writes E2E test scenarios. 
    Use the following pieces of context to do actions at the end.
    {context}

    Action: {question}
    """

    query = f"""
    Create 30 E2E business logic test scenarios based on document,
    and choose only {TEST_SCENARIO_COUNT} important test scenarios related to users in Project Manager's perspective.

    Ignore configuration files such as webpack, package.json, etc. Embed business-logic-related files only.
    
    {TEST_SCENARIO_COUNT} E2E detail test cases(from 30 generated E2E tests) in BULLET POINTS:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    index = VectorstoreIndexCreator().from_loaders([loader])

    retriever = index.vectorstore.as_retriever()

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=retriever,
                                     chain_type_kwargs=chain_type_kwargs)

    test_scenario_file_path = f"./generated/{repo_name}/test-scenario.txt"
    os.makedirs(os.path.dirname(test_scenario_file_path), exist_ok=True)
    with open(test_scenario_file_path, "w") as f:
        f.write(qa.run(query))

# This function is almost identical as the one in yaml_gen.py
# However, the model used is "gpt-3.5-turbo"
def generate_test_codes():
    repo_name = WEB_NAME
    file_summary_path = f"./generated/{repo_name}/file_summary"
    test_scenario_file_path = f"./generated/{repo_name}/test-scenario.txt"

    if not os.path.exists(test_scenario_file_path):
        raise Exception(
            'test_scenario_file_path not exists. Please run test-scenario-gen to generate test-scenario.txt.')

    if os.path.exists(file_summary_path):
        loader = DirectoryLoader(file_summary_path, loader_cls=TextLoader)
    else:
        raise Exception(
            'file_summary does not exist. Please run summary-gen to generate file_summary.')

    with open(test_scenario_file_path, "r") as f:
        test_scenarios = f.read()

        for i in range(int(TEST_SCENARIO_COUNT)):
            prompt_template = """You are a Software Engineer who writes test codes. 
            Your language/framework preference is Javascript(Node.js, Jest).
            Use the following pieces of context to do actions at the end.
            {context}

            Action: {question}
            """

            query = f"""
            Create E2E test code for ONLY the {i + 1}th business logic of below test scenario document.
            E2E test code should be in Javscript language which works in Node.js environment.
            At the beggining of the code, test scenario must be embedded in comment section.

            [test-scenario.txt]
            {test_scenarios}

            Professional & Detail E2E test code in Javascript:
            """

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=[
                    "context", "question"]
            )

            index = VectorstoreIndexCreator().from_loaders([loader])

            retriever = index.vectorstore.as_retriever()

            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=retriever,
                                             chain_type_kwargs=chain_type_kwargs)

            test_code_file_path = f"./generated/{repo_name}/test-codes/test-code-{i+1}.js"
            os.makedirs(os.path.dirname(test_code_file_path), exist_ok=True)
            with open(test_code_file_path, "w") as f:
                f.write(qa.run(query)
                        .replace("```javascript", "").replace("```", ""))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_mode", "-m", type=str, required=True,
                        help="Generate mode - summary-gen, fetch-elems, fetch-sect.")

    args = parser.parse_args()

    if args.generate_mode == "summary-gen":
        generate_file_summary_yaml()
    elif args.generate_mode == "fetch-elems":
        fetch_file_elements_yaml()
    elif args.generate_mode == "fetch-sect":
        fetch_file_sects_yaml()
    elif args.generate_mode == "test-scenario-gen":
        create_test_scenario()
    elif args.generate_mode == "test-gen":
        generate_test_codes()
    else:
        print("generate_mode(-m) option should be one of summary-gen, fetch-elems, fetch-sect. Please try again.")
