from langchain.document_loaders.merge import MergedDataLoader
from langchain.docstore.document import Document

from typing import List
import re


def parse_intermediate_steps(_list):
    _steps = []
    for n in range(len(_list)):
        i = _list[n]
        i_str = f"Step {n+1}: {i[0].tool}\n"
        i_str += f"> {i[0].tool_input}\n"
        i_str += f"< {i[0].log}\n"
        i_str += f"# {i[1]}\n"
        _steps.append(i_str)
    return "\n".join(_steps)


def merge_doc_loader(_list: List[Document]):
    loader_all = MergedDataLoader(loaders=_list)
    docs_all = loader_all.load()
    return docs_all


def calc_token_cost(_tc: list):
    total_tokens = 0
    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    # 对于列表中的每个字符串，使用正则表达式解析出需要的数字
    for entry in _tc:
        tokens_match = re.search(r"Tokens: (\d+)", entry)
        prompt_match = re.search(r"Prompt (\d+)", entry)
        completion_match = re.search(r"Completion (\d+)", entry)
        cost_match = re.search(r"Cost: \$([\d.]+)", entry)
        if tokens_match:
            total_tokens += int(tokens_match.group(1))
        if prompt_match:
            total_prompt += int(prompt_match.group(1))
        if completion_match:
            total_completion += int(completion_match.group(1))
        if cost_match:
            total_cost += float(cost_match.group(1))
    # 格式化并输出结果
    output = f"Tokens: {total_tokens} = (Prompt {total_prompt} + Completion {total_completion}) Cost: ${total_cost:.5f}"
    return output

