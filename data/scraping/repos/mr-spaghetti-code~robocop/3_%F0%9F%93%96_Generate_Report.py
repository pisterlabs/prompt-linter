import anthropic
import datetime
import json
import os
import pickle
import tempfile
from lxml import etree

from prompts.claude import prompts
import streamlit as st
from storage3 import create_client

from streamlit.logger import get_logger
from langchain.chains import LLMChain
from langchain.chat_models import ChatAnthropic
from langchain.document_loaders import GitLoader
from langchain.llms import Anthropic
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Report", page_icon="📖")

st.markdown(
    """This section pulls raw code directly from Github instead of finding relevant embeddings then generates a report.
    """
)

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

# if "vulnerabilities_to_find" not in st.session_state:
#     st.session_state["vulnerabilities_to_find"] = []


os.environ['ANTHROPIC_API_KEY'] = st.session_state["anthropic_api_key"] if st.session_state["settings_override"] else st.secrets.anthropic_api_key

st.markdown("# Report")

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

render_output = st.checkbox('Render output', value=True)

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
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", prefix="report_") as fp:
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
            file_filter=lambda file_path: file_path.endswith(".sol")
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

def get_code_summary(code):
    chain = LLMChain(llm=llm, prompt=prompts.USER_TEMPLATE_PROVIDE_SUMMARY)
    response = chain.run({
        'code': code
        })
    return response


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
            logger.info(contracts)
            contract_names = [contract.metadata["file_path"] for contract in contracts]
            st.session_state["contract_names"] = contract_names


st.header("Contracts")

reports_to_generate = st.multiselect(
    "Pick the smart contracts you want to generate reports for.",
    st.session_state["contract_names"]
)

st.session_state["reports_to_generate"] = reports_to_generate

vulnerabilities_to_find = st.multiselect(
    "Pick the vulnerabilities to look for.",
    list(prompts.VULNERABILITIES.keys())
)

generated_reports = []

llm = Anthropic(
    model="claude-2",
    temperature=0,
    max_tokens_to_sample=1024,
    verbose=True
)

output_txt = ""

if st.button("Generate Reports"):
    status = st.info(f'Generating reports', icon="ℹ️")
    current_date = datetime.date.today()
    output_txt += f"# Robocop Audit Report for \n{github_url}\n\nDate: {current_date}\n\n"
    formatted_files = [f"* {report}" for report in st.session_state["reports_to_generate"]]
    scope = "\n".join(formatted_files)
    output_txt += scope + "\n"
    for report in st.session_state["reports_to_generate"]:
        st.info(f'Generating report for {report}', icon="ℹ️")
        summary = ''
        gen_report = {}
        gen_report[report] = {}
        gen_report[report]["file"] = report
        
        output_txt += f"# File Analyzed: {report}\n"
        with st.spinner('Retrieving code...'):
            code = filter_by_name(report.split("/")[-1])[0].page_content
            num_tokens = anthropic.count_tokens(code)
            gen_report[report]['code'] = code
            gen_report[report]['num_tokens_code'] = num_tokens
            logger.info(f"Processing code:\n{code}")
            status.info(f'Retrieved code for {report} - Processing {num_tokens} tokens.', icon="ℹ️")
        with st.spinner('Getting summary...'):
            response = get_code_summary(code)
            logger.info(f"RESPONSE RECEIVED\n*********\n{response}")
            gen_report[report]['summary'] = response
            output_txt += response + "\n"
            if render_output:
                st.write(response)
        for bug_type in vulnerabilities_to_find:
            with st.spinner(f'Scanning for {bug_type} bugs...'):
                formatted_task = prompts.USER_TEMPLATE_TASK.format(
                    type=bug_type, 
                    description=prompts.VULNERABILITIES[bug_type]["description"], 
                    examples=prompts.VULNERABILITIES[bug_type]["description"])
                
                chain = LLMChain(llm=llm, prompt=prompts.USER_TEMPLATE_WITH_SUMMARY)
                response = chain.run({
                    "smart_contract_name": report,
                    "summary": summary,
                    "code": code,
                    "task": formatted_task
                    })
                
                review_chain = LLMChain(llm=llm, prompt=prompts.REVIEWER_TASK_TEMPLATE)

                logger.info(f"RESPONSE RECEIVED\n*********\n{response}")

                resp_parsed = etree.fromstring(response.strip(), parser=parser)
                
                ui_outputs = []
                found_bugs = []

                for vulnerability in resp_parsed:
                    # logger.info(vulnerability[0].text)
                    # logger.info(vulnerability.text)

                    try:
                        vulnerability_instance = {}
                        vulnerability_instance["description"] = vulnerability[0].text
                        vulnerability_instance["severity"] = vulnerability[1].text
                        vulnerability_instance["impact"] = vulnerability[2].text
                        vulnerability_instance["recommendation"] = vulnerability[3].text
                        print(vulnerability_instance)
                        ui_output = f"""### Description\n\n{vulnerability_instance["description"]}\n\n### Severity\n\n{vulnerability_instance["severity"]}\n\n### Impact\n\n{vulnerability_instance["impact"]}\n\n### Recommendation\n\n{vulnerability_instance["recommendation"]}\n\n"""

                        gen_report[report]['bugs'] = {
                            bug_type : found_bugs
                        }
                        review = review_chain.run({
                            "report": ui_output,
                            "code": code
                        })

                        ui_output += "---\n### Expert Reviewer\n" + review
                        ui_outputs.append(ui_output)
                        found_bugs.append(vulnerability_instance)
                        print(ui_output)
                    except:
                        logger.info("No vulnerabities found")
                    
                        

                header = f"## Analysis results for {bug_type} vulnerabilities \n\n"
                output_txt += header
                st.write(header)

                try:
                    for output in ui_outputs:
                        logger.info(output)
                        if render_output: st.write(output)
                        output_txt += output + "\n"
                except:
                    st.write("N/A")
            generated_reports.append(gen_report)
    logger.info(generated_reports)
    json_obj = json.dumps(generated_reports)
    status.success("Done!")
    st.balloons()

    if st.button("Save Report"):
        save_report(project_name)

    
    st.download_button(
        label="Download data as JSON",
        data=json_obj,
        file_name='report_findings.json',
        mime='application/json',
    )

    st.download_button(
        label="Download data as Text (markdown)",
        data=output_txt,
        file_name='report_findings.md',
        mime='text/plain',
    )