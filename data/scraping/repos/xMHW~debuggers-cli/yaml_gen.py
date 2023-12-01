import os
import argparse
from dotenv import load_dotenv

from langchain import LLMChain
from langchain.document_loaders import GitLoader, DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()


TEST_SCENARIO_COUNT = os.getenv("TEST_SCENARIO_COUNT")
GIT_REPOSITORY = os.getenv("GIT_REPOSITORY")
REPO_NAME = GIT_REPOSITORY.split('/')[-1].split('.')[0]
GENERATED_DIR = f"./generated/{REPO_NAME}"
TEST_CODES_DIR = f"{GENERATED_DIR}/test-codes"
FILE_SUMMARY_DIR = f"{GENERATED_DIR}/file_summary"
TEST_SCENARIO_FILE_PATH = f"{GENERATED_DIR}/test-scenario.txt"

OPENAI_MODELS = {
    "DAVINCI": "text-davinci-003",
    "GPT-3": "gpt-3.5-turbo",
    "GPT-4": "gpt-4"
}


def generate_test_codes():
    if not os.path.exists(TEST_SCENARIO_FILE_PATH):
        raise Exception(
            'test_scenario_file_path not exists. Please run test-scenario-gen to generate test-scenario.txt.')

    if os.path.exists(FILE_SUMMARY_DIR):
        loader = DirectoryLoader(FILE_SUMMARY_DIR, loader_cls=TextLoader)
    else:
        raise Exception(
            'file_summary does not exist. Please run summary-gen to generate file_summary.')

    with open(TEST_SCENARIO_FILE_PATH, "r") as f:
        test_scenarios = f.read()

        for i in range(TEST_SCENARIO_COUNT):
            prompt_template = """You are a Software Engineer who writes test codes. 
            Your language/framework preference is Javascript(Node.js, Jest).
            Use the following pieces of context to do actions at the end.
            {context}

            Action: {question}
            """

            query = f"""
            Create E2E test code for {i + 1}th business logic of below test scenario document.
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
            qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name=OPENAI_MODELS["GPT-4"]), chain_type="stuff", retriever=retriever,
                                             chain_type_kwargs=chain_type_kwargs)

            test_code_file_path = f"{TEST_CODES_DIR}/test-code-{i+1}.js"
            os.makedirs(os.path.dirname(test_code_file_path), exist_ok=True)
            with open(test_code_file_path, "w") as f:
                f.write(qa.run(query)
                        .replace("```javascript", "").replace("```", ""))


def create_test_scenario():
    if os.path.exists(FILE_SUMMARY_DIR):
        loader = DirectoryLoader(FILE_SUMMARY_DIR, loader_cls=TextLoader)
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
    Each test scenario should have the following parts: Title, Description, and Expected Result.

    Ignore configuration files such as webpack, package.json, etc. Embed business-logic-related files only.
    
    {TEST_SCENARIO_COUNT} E2E detail test cases(from 30 generated E2E tests) in BULLET POINTS:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    index = VectorstoreIndexCreator().from_loaders([loader])

    retriever = index.vectorstore.as_retriever()

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name=OPENAI_MODELS["GPT-4"]), chain_type="stuff", retriever=retriever,
                                     chain_type_kwargs=chain_type_kwargs)

    os.makedirs(os.path.dirname(TEST_SCENARIO_FILE_PATH), exist_ok=True)
    with open(TEST_SCENARIO_FILE_PATH, "w") as f:
        f.write(qa.run(query))


def generate_file_summary_yaml():
    repo_path = f"./example_repo/{REPO_NAME}"
    supported_extensions = (".js", ".ts", ".tsx")

    if os.path.exists(repo_path):
        try:
            loader = GitLoader(repo_path=repo_path, branch="master",
                            file_filter=lambda file_path: file_path.endswith(supported_extensions))
            data = loader.load()
        except:
            loader = GitLoader(repo_path=repo_path, branch="main",
                            file_filter=lambda file_path: file_path.endswith(supported_extensions))
            data = loader.load()
    else:
        try:
            loader = GitLoader(clone_url=GIT_REPOSITORY,
                            repo_path=repo_path, branch="master",
                            file_filter=lambda file_path: file_path.endswith(supported_extensions))
            data = loader.load()
        except:
            loader = GitLoader(clone_url=GIT_REPOSITORY,
                            repo_path=repo_path, branch="main",
                            file_filter=lambda file_path: file_path.endswith(supported_extensions))
            data = loader.load()

    print("Data Load Finished!")

    query = """
    You are a professional Software Engineer who has scarce knowledge about E2E testing and file summarization.
    Create a definition file that summarizes and explains about file in yaml format.
    In the generated yaml format text, it should contain important information about the file.
    
    [File : {file_path}]
    {page_content}

    File Summary in yaml Format:
    """
    template = PromptTemplate(template=query, input_variables=[
                              'file_path', 'page_content'])

    openai = OpenAI(model_name="gpt-3.5-turbo")
    chain = LLMChain(prompt=template, llm=openai)

    for file in data:
        page_content = file.page_content
        file_path = file.metadata["file_path"]
        print(file_path)

        gen_file_path = f"{FILE_SUMMARY_DIR}/{file_path}.yaml"

        os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)
        with open(gen_file_path, "w") as f:
            f.write(
                chain.run({'file_path': file_path, 'page_content': page_content}))
            f.close()
    print("File Summary Generation Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_mode", "-m", type=str, required=True,
                        help="Generate mode - summary-gen, test-scenario-gen, test-gen")

    args = parser.parse_args()

    if args.generate_mode == "summary-gen":
        generate_file_summary_yaml()
    elif args.generate_mode == "test-scenario-gen":
        create_test_scenario()
    elif args.generate_mode == "test-gen":
        generate_test_codes()
    else:
        print("generate_mode(-m) option should be one of summary-gen, test-scenario-gen, test-gen. Please try again.")
