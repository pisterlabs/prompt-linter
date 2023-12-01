import re
from pathlib import Path

from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import GitLoader
from langchain.agents import tool, AgentExecutor, ZeroShotAgent


def load_repo(repo: str, branch: str = "main") -> list:
    path = Path("repos") / repo
    url = f"https://github.com/{repo}" if not path.exists() else None
    loader = GitLoader(
        repo_path=path,
        clone_url=url,
        branch=branch,
        file_filter=lambda x: not x.startswith(".")
    )
    return loader.load()


@tool
def get_file_content(repo_file_path: str) -> str:
    """Get content of specific file in repo. Input be like `user/repo:file_path"""
    repo, file_path = repo_file_path.split(":")
    file = Path("repos") / repo / file_path
    return file.read_text()[:2000] if file.is_file() else "Not found"


@tool
def get_repo_info(repo: str) -> str:
    """Get files tree and README of the repo"""
    files = load_repo(repo)
    tree = [
        f.metadata["file_path"]
        for f in files
        if not re.match("^\.|(?:tests?)", f.metadata["file_path"])
    ]
    readme = get_file_content(f"{repo}:README.md")
    info = "Repo: {}\n\nFiles:\n{}\n\nReadme:\n{}".format(repo, "\n".join(tree), readme[0])
    return info[:2000]


def get_judge(hackathon_info: dict, model_config: dict = {}, verbose: bool = False):
    llm = ChatOpenAI(**model_config)
    tools = [get_repo_info, get_file_content]
    system_prompt = hub.pull("xleven/ai-hackathon-judge")
    prompt = system_prompt.partial(
        tool_strings="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools]),
        **hackathon_info,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=chain, allowed_tools=[tool.name for tool in tools])
    judge = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)
    return judge
