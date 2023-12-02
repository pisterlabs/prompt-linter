import anthropic
import datetime
import json
import os
import pickle
import tempfile
from lxml import etree
import re


from prompts.claude import prompts
import streamlit as st
from storage3 import create_client

from streamlit.logger import get_logger
from langchain.chains import LLMChain
from langchain.chat_models import ChatAnthropic
from langchain.document_loaders import GitLoader
from langchain.llms import Anthropic


from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

st.set_page_config(page_title="Summarize Codebase", page_icon="📖")

logger = get_logger(__name__)
parser = etree.XMLParser(recover=True)
headers = {"apiKey": st.secrets.supabase_key, "Authorization": f"Bearer {st.secrets.supabase_key}"}
storage_client = create_client(st.secrets.supabase_url, headers, is_async=False)

if "anthropic_api_key" not in st.session_state:
    st.session_state["anthropic_api_key"] = ''

if "raw_code" not in st.session_state:
    st.session_state["raw_code"] = []

if "settings_override" not in st.session_state:
    st.session_state["settings_override"] = []

if "contract_names" not in st.session_state:
    st.session_state["contract_names"] = []

if "reports_to_generate" not in st.session_state:
    st.session_state["reports_to_generate"] = []

os.environ['ANTHROPIC_API_KEY'] = st.session_state["anthropic_api_key"] if st.session_state["settings_override"] else st.secrets.anthropic_api_key

st.markdown("# Summarize entire codebase")

def get_github_folders():
    bucket = storage_client.from_('repo_contents')
    res = bucket.list()
    proj_list = ["New project"]
    return proj_list + [dir["name"] for dir in res]

existing_github_folders = get_github_folders()

project_option = st.selectbox(
    'Pick from existing Github repo?',
    existing_github_folders
    )

project_name = st.text_input(
    label="Enter a project name. This will be used to save the report.")

github_url = st.text_input(
    label="Enter the URL of a _public_ GitHub repo")

commit_branch = st.text_input(label="Enter the commit ID (optional) or branch (default:main",
    value="master or main or your commit ID")

load_markdown = st.checkbox(
    label="Load markdown files alongside contracts",
    value=True
)

def save_github_files(data):
    bucket = storage_client.from_('repo_contents')
    file = open('test.p', 'wb')
    pickle.dump(data, file)
    name = f"{project_name}/github.pkl"
    res = bucket.upload(
        name,
        os.path.abspath("test.p")
    )
    st.write(res)
    logger.info("Data saved successfully to Supabase.")
    file.close()

def get_github_files(project_name):
    logger.info(f"Fetching Github files from: {project_name}")
    bucket = storage_client.from_('repo_contents')
    name = f"{project_name}/github.pkl"
    with open("test.p", 'wb+') as f:
        res = bucket.download(name)
        f.write(res)
    data = pickle.load( open( "test.p", "rb" ) )
    logger.info("Data loaded successfully from Supabase.")
    return data

def save_report(project_name):
    bucket = storage_client.from_('reports')
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", prefix="summary") as fp:
        fp.write(output_txt)
        
        print(fp.name.split("/")[-1])
        path = project_name + "/" + fp.name.split("/")[-1]
        res = bucket.upload(
                path,
                os.path.abspath(fp.name)
        )
        logger.info(res)
        st.write(res)

def check_if_dump_exists(project_name):
    logger.info("Check if file exists...")
    bucket = storage_client.from_('repo_contents')
    file = "github.pkl"
    res = bucket.list(project_name)
    exists = any(file in files.values() for files in res)
    logger.info(f"File exists: {exists}")
    return exists

def load_text(clone_url, project_name):
    # loader = GitLoader(repo_path="./juice-buyback/")
    if project_option != "New project":
        project_name = project_option
    exists = check_if_dump_exists(project_name)
    if exists:
        data = get_github_files(project_name)
    else:
        loader = GitLoader(
            clone_url=clone_url,
            repo_path=tmpdirname,
            branch=commit_branch,
            file_filter=lambda file_path: file_path.endswith((".sol",".md"))
        )
        data = loader.load()
        save_github_files(data)
    st.session_state["raw_code"] = data
    return data

def filter_by_type(file_type):
    filtered_text = list(filter(lambda doc: (doc.metadata['file_type'] == file_type), st.session_state["raw_code"]))
    return filtered_text

def filter_by_name(name):
    filtered_text = list(filter(lambda doc: (doc.metadata['file_name'] == name), st.session_state["raw_code"]))
    return filtered_text

button = st.button("Analyze")

if button:
    status = st.info(f'Pulling from {github_url}', icon="ℹ️")
    with st.spinner('Processing...'):
        with tempfile.TemporaryDirectory() as tmpdirname:
            logger.info(f'Created temporary directory: {tmpdirname}')

            status.info("Loading data")

            texts = load_text(
                clone_url=github_url,
                project_name=project_name)

            status.info("Data loaded")

            logger.info("Data retrieved")

            contracts = filter_by_type(".sol")

            if load_markdown: 
                contracts +=  filter_by_type(".md")

            logger.info([type(contracts), len(contracts)])
            # logger.info(contracts)
            contract_names = [contract.metadata["file_path"] for contract in contracts]
            st.session_state["contract_names"] = contract_names


st.header("Contracts")

reports_to_generate = st.multiselect(
    "Pick the smart contracts you want to generate reports for.",
    st.session_state["contract_names"]
)

st.session_state["reports_to_generate"] = reports_to_generate

generated_reports = []

llm = Anthropic(
    temperature=0,
    max_tokens_to_sample=1024*4,
    # max_tokens_limit=80000,
    verbose=True
)

output_txt = ""

if st.button("Generate Summary"):
    current_date = datetime.date.today()
    output_txt += f"# Robocop Audit Report for \n{github_url}\n\nDate: {current_date}\n\n"
    formatted_files = [f"* {report}" for report in st.session_state["reports_to_generate"]]
    scope = "\n".join(formatted_files)
    output_txt += scope + "\n"

    all_code = ""
    status = st.info(f'Generating report for \n{scope}', icon="ℹ️")
    for report in st.session_state["reports_to_generate"]:
        code = filter_by_name(report.split("/")[-1])[0].page_content
        all_code += f"\n\n{report}\n\nCode:\n\n```{code}```"
    logger.info(all_code)
    logger.info("Generating code base summary...")
    with st.spinner('Generating code base summary...'):
        chain = LLMChain(llm=llm, prompt=prompts.USER_TEMPLATE_PROVIDE_SUMMARY_ENTIRE_CODEBASE)
        response = chain.run({
            'repo_name': project_option,
            'code': all_code
            })
        st.write(response)
        output_txt += response
    
        # summary = ''
        # gen_report = {}
        # gen_report[report] = {}
        # gen_report[report]["file"] = report
        
        # output_txt += f"# File Analyzed: {report}\n"
        # with st.spinner('Retrieving code...'):
        #     code = filter_by_name(report.split("/")[-1])[0].page_content
        #     num_tokens = anthropic.count_tokens(code)
        #     gen_report[report]['code'] = code
        #     gen_report[report]['num_tokens_code'] = num_tokens
        #     logger.info(f"Processing code:\n{code}")
        #     status.info(f'Retrieved code for {report} - Processing {num_tokens} tokens.', icon="ℹ️")
        # with st.spinner('Getting summary...'):
        #     chain = LLMChain(llm=llm, prompt=prompts.USER_TEMPLATE_PROVIDE_SUMMARY)
        #     response = chain.run({
        #         'code': code
        #         })
        #     summary = response
        #     logger.info(f"RESPONSE RECEIVED\n*********\n{response}")
        #     gen_report[report]['summary'] = response
        #     output_txt += response + "\n"
        #     if render_output:
        #         st.write(response)
        # with st.spinner('Generating unit tests...'):
        #     chain = LLMChain(llm=llm, prompt=prompts.USER_TEMPLATE_WITH_SUMMARY)
        #     response = chain.run({
        #         "smart_contract_name": report,
        #         "summary": summary,
        #         "code": code,
        #         "task": prompts.CONTEXT_TEMPLATE_UNIT_TESTS
        #         })
        #     logger.info(f"RESPONSE RECEIVED\n*********\n{response}")
        #     resp_parsed = etree.fromstring(response.strip(), parser=parser)
        #     logger.info(resp_parsed)
        #     ui_outputs = []
        #     tests_produced = []
        #     for unit_test in resp_parsed:
        #         try:
        #             unit_test_instance = {}
        #             unit_test_instance["description"] = unit_test[0].text
        #             unit_test_instance["code"] = unit_test[1].text
        #             ui_output = f"""### Description\n\n{unit_test_instance["description"]}\n\n### Unit Test\n\n{unit_test_instance["code"]}"""
        #             ui_outputs.append(ui_output)
        #             tests_produced.append(unit_test_instance)

        #         except:
        #             logger.info("No unit tests found")
        #     for output in ui_outputs:
        #         logger.info(output)
        #         if render_output:
        #             st.write(output)
        #             st.divider()



    # logger.info(generated_reports)
    # json_obj = json.dumps(generated_reports)
    logger.info("Done!")
    status.success("Done!")
    st.balloons()

    if st.button("Save Summary"):
        save_report(project_name)

    st.download_button(
        label="Download data as Text (markdown)",
        data=output_txt,
        file_name='repo_summary.md',
        mime='text/plain',
    )