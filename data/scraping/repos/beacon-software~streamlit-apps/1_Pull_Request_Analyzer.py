import requests
import streamlit as st
import unidiff
from github import Github, PullRequest, File

from langchain.llms import OpenAIChat

st.set_page_config("Pull Requests")

openai_token = st.secrets["openai_token"]
github_token = st.secrets["github_token"]
st.session_state["repository"] = "KimleyHorn/SigOpsMetrics"

llm = OpenAIChat(openai_api_key=openai_token)

@st.cache_data
def call_openai_chat(text: str):
    return llm(text)

@st.cache_data
def get_pull_requests(repository_name: str):
    return requests.get(f"https://api.github.com/repos/{repository_name}/pulls").json()

@st.cache_data
def get_git_info(repository_name: str, pull_request: int):
    g = Github(login_or_token=github_token)
    repo = g.get_repo(st.session_state["repository"])
    pr = repo.get_pull(pull_request)
    return [
        requests.get(pr.diff_url).text,
        pr.title,
        pr.body,
    ]

if "pull_requests" not in st.session_state:
    st.session_state["pull_requests"] = get_pull_requests(st.session_state["repository"])
    st.session_state["selected_pull_request_title"] = st.session_state["pull_requests"][0]["title"]

with st.sidebar:
    st.header("Beacon - Pull Request Analyzer")
    st.text_input("Repository URL", value=st.session_state["repository"], disabled=True)

    st.selectbox("Pull Request", options=[x["title"] for x in st.session_state["pull_requests"]], key="selected_pull_request_title")

pull_request = next(pr for pr in st.session_state["pull_requests"] if pr["title"] == st.session_state["selected_pull_request_title"])

patch, title, body = get_git_info(st.session_state["repository"], pull_request["number"])

st.header("Pull Request Summary")
st.subheader(f"# {pull_request['number']} - {title}")
st.write(body)
st.select_slider("Level Of Technical Detail", options=["More Technical", "Slightly Technical", "Non-Technical"], key="technical_detail")

[col1, col2, col3] = st.columns(3)

patchset = unidiff.PatchSet(patch)

with col1:
    st.metric("Number of Files Changed", value=len(patchset))
with col2:
    st.metric("Change Type", "Feature")
with col3:
    st.metric("Status", pull_request['state'].capitalize())

if st.session_state["technical_detail"] == "More Technical":

    with st.expander("View Diff"):
        st.code(patch)
    st.write("**What changes were made by this pull request?**")
    st.write(call_openai_chat(f"""
    Act as a software engineer reviewing a pull request.

    You are provided with a pull request in patch format
    Each patch entry has a commit message in the subject line followed by the code changes (diffs) in a unidiff format

    {patch}

    As an engineer, your tasks are:
    - Describe what changes were made in the pull request
    """))
    st.write("**Does this pull request meet the quality standard?**")
    st.write(call_openai_chat(f"""
    Act as a software engineer reviewing a pull request.

    You are provided with a pull request in patch format
    Each patch entry has a commit message in the subject line followed by the code changes (diffs) in a unidiff format

    {patch}

    As an engineer, your tasks are:
    - Review the changes for quality & correctness
    - Identify any bugs or issues that you see
    - Give your opinion on whether or not this code can be merged
    """))
elif st.session_state["technical_detail"] == "Slightly Technical":
    st.write("**What changes were made by this pull request?**")
    st.write(call_openai_chat(f"""
    Act as a software engineer explaining a pull request to a non-technical member of your team.

    You are provided with a pull request in patch format
    Each patch entry has a commit message in the subject line followed by the code changes (diffs) in a unidiff format

    {patch}

    As an engineer, your tasks are:
    - Describe what changes were made in the pull request
    """))
    st.write("**Does this pull request meet the quality standard?**")
    st.write(call_openai_chat(f"""
    Act as a software engineer reviewing a pull request.

    You are provided with a pull request in patch format
    Each patch entry has a commit message in the subject line followed by the code changes (diffs) in a unidiff format

    {patch}

    As an engineer, your tasks are:
    - Review the changes for quality & correctness
    - Identify any bugs or issues that you see
    - Do not use code or technical terms in your response
    """))
else:
    st.write("**What type of ticket is this?**")
    st.write(call_openai_chat(f"""
    You are an engineer trying to describe the following ticket.

    Title: {title}
    Description: {body}

    Your task:
    Decide whether this pull request is related to new features, bug fixes, system upgrades, or refactoring.
    If it is a new feature, describe in one sentence what the feature is and respond with "FEATURE: description".
    If it is a bug fix, describe in one sentence what the bug is and respond with "BUG: description".
    If it is a system upgrade, describe in one sentence what the upgrade is and respond with "UPGRADE: description".
    If it is a refactoring, describe in one sentence what the refactoring is and respond with "REFACTOR: description".
    """))

    st.write("**What changes were made by this pull request?**")
    st.write(call_openai_chat(f"""
    You are an engineer describing a pull request to a non-technical member of your team.

    You are provided with the title and description of the pull request.

    Title: {title}
    Description: {body}

    Your task:
    Describe what you did in the pull request.
    Describe why someone would want to do what you did in the pull request.
    Do not use "I", "we", or "you" in your description.
    """))

    st.write("**What should you ask the engineer about this pull request?**")
    st.write(call_openai_chat(f"""
    You are a non-technical product manager 

    You are provided with the title and description of the pull request.

    Title: {title}
    Description: {body}

    Your task:
    Provide a list of 2-3 questions you would ask the engineer about this pull request.
    """))