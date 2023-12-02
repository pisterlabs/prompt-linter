from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Union, Tuple
import json
import requests
import os

from . import codextract
from .openai import OPENAI_API_KEY, completion, extract_markdown_codes
from .codextract import ExtractedInfo
from .prompts import PROMPTS
from .pipeline import workspace_for_repo, workspace_by_name, workspaces_list

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'static')), name="static")


class UrlInput(BaseModel):
    url: str


class FileList(BaseModel):
    files: List[str]


class TreeNode(BaseModel):
    name: str
    type: str
    data: Union[List, None]
    children: List[Union['TreeNode', None]] = Field(default_factory=list)


class ListData(BaseModel):
    items: List[TreeNode]


TreeNode.update_forward_refs()


def create_tree_node(path: str, is_directory: bool) -> Union[TreeNode, None]:
    if is_directory:
        children = []
        for item in os.listdir(path):
            child_path = os.path.join(path, item)
            child_node = create_tree_node(child_path, os.path.isdir(child_path))
            if child_node:
                children.append(child_node)
        if len(children) != 0:
            return TreeNode(name=os.path.basename(path), type="directory", children=children)
        return None
    if os.path.splitext(path)[1] == ".sol":
        return TreeNode(name=os.path.basename(path), type="file")


def directory_to_list_data(dir_path: str) -> ListData:
    if not os.path.isdir(dir_path):
        raise ValueError(f"The provided path is not a directory: {dir_path}")

    root_items = []
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        node = create_tree_node(item_path, os.path.isdir(item_path))
        if node:
            root_items.append(node)
    return ListData(items=[TreeNode(name=os.path.basename(dir_path), type="directory", children=root_items)])


def create_contract_nodes(info:ExtractedInfo, lines) -> ListData:
    contracts = []
    for c in info.contracts:
        children = []
        # print('\t', 'library' if c.is_library else 'contract', c.name, c.line_start, c.line_end)
        # for _ in c.events:  print('\t\t', _)
        # for _ in c.using:   print('\t\t', _)
        # for _ in c.types:   print('\t\t', _)
        for u in c.uncaught:
            name = u.name
            data = [u.line_start, u.line_end]
            children.append(TreeNode(name=name, type="uncaught", data=data, children=[]))
        for e in c.events:
            name = e.signature()
            data = [e.line_start, e.line_end]
            children.append(TreeNode(name=name, type="event", data=data, children=[]))
        for t in c.types:
            children.append(TreeNode(name=t, type="type", data=data, children=[]))
        for f in c.functions:
            name = f.signature()
            data = [f.line_start, f.line_end]
            children.append(TreeNode(name=name, type="function", data=data, children=[]))
        for m in c.modifiers:
            name = m.name
            data = [m.line_start, m.line_end]
            children.append(TreeNode(name=name, type="modifier", data=data, children=[]))
        contracts.append(TreeNode(name=c.name, type="contract", data=[c.line_start, c.line_end], children=children))
    return ListData(items=contracts)


@app.get("/")
async def serve_index_html():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'static', 'index.html'))


@app.get("/favicon.ico")
async def serve_index_html():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'static', 'favicon.ico'))


@app.post("/message/")
async def chatbot_response(message: str):
    response = requests.post(
        "https://api.openai.com/v1/engines/davinci-codex/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "prompt": message,
            "max_tokens": 150,
            "n": 1,
            "stop": None,
            "temperature": 0.5,
        },
    )
    response.raise_for_status()
    return {"response": response.json()["choices"][0]["text"]}


@app.get("/repo/{workspace_id}")
async def workspace(workspace_id:str):
    p = workspace_by_name(workspace_id)
    return dict(workspace_id=workspace_id, data=directory_to_list_data(p.path('repo')))


@app.post("/repo.clone/")
async def repo_clone(repo: UrlInput):
    sh, p = workspace_for_repo(repo.url)
    p.git_clone()
    return dict(workspace_id=sh, data=directory_to_list_data(p.path('repo')))


@app.get("/prompts")
async def prompts():
    return {str(k): {
        'description': v.description,
        'cats': v.cats,
        'model': v.model,
    } for k, v in PROMPTS.items()}


def _collect_files(workspace_id, file_list):
    ws = workspace_by_name(workspace_id)
    collected_lines = []
    for file in file_list:
        with ws.open('r', file) as f:
            collected_lines += f.readlines()
    return collected_lines


@app.post("/source/{workspace_id}")
async def source(workspace_id, file_list: FileList):    
    collected_lines = _collect_files(workspace_id, file_list.files)
    info = codextract.analyze_lines("CombinedContracts", collected_lines)
    contracts_list = create_contract_nodes(info, collected_lines)
    joined_lines = "".join(collected_lines)
    return {"contracts": contracts_list, "lines": joined_lines}


class AnalyzeInput(BaseModel):
    prompt: str
    files: List[str]
    lines: List[str]


@app.post("/analyze/{workspace_id}")
async def analyze(workspace_id:str, data:AnalyzeInput):
    w = workspace_by_name(workspace_id)
    p = PROMPTS[data.prompt]
    v = set()
    for _ in data.lines:
        a, b = _.split(',')
        v = v.union(set(range(int(a),int(b)+1)))
    analyze_text = "".join([y for z, y in enumerate(_collect_files(workspace_id, data.files)) if z in v])
    info = completion(p, {'CODE': analyze_text}, cache_ctx={
        'file_list': data.files,
        'input_lines': data.lines,
        'workspace_id': workspace_id,
        'parent_cid': None
    })
    ajpath = 'analyzer', info['cid'] + '.json'
    if not w.exists(*ajpath):
        info['code_outputs'] = extract_markdown_codes(info['result'])           
        info['run'] = w.run_codes(info['cid'], info['code_outputs'])
        with w.open('w', *ajpath) as handle:
            handle.write(json.dumps(info))
        return info
    with w.open('r', *ajpath) as handle:
            return json.load(handle)


@app.get('/workspaces')
async def workspaces():
    return {p.guid: p.config() for p in map(workspace_by_name, workspaces_list())}
        
