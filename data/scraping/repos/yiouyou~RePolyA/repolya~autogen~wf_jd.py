from repolya._const import WORKSPACE_AUTOGEN, AUTOGEN_JD
from repolya._log import logger_yj

from repolya.autogen.organizer import (
    Organizer,
    ConversationResult,
)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)
from langchain.schema import StrOutputParser
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import WebBaseLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

from repolya.toolset.tool_langchain import (
    bing,
    ddg,
    google,
)
from repolya.toolset.util import calc_token_cost
from repolya.rag.digest_dir import (
    calculate_md5,
    dir_to_faiss_OpenAI,
)
from repolya.rag.doc_loader import clean_txt
from repolya.rag.digest_urls import (
    urls_to_faiss_OpenAI,
    urls_to_faiss_HuggingFace,
)
from repolya.rag.vdb_faiss import (
    get_faiss_OpenAI,
    get_faiss_HuggingFace,
)
from repolya.rag.qa_chain import (
    qa_vdb_multi_query,
    qa_vdb_multi_query_textgen,
    qa_with_context_as_go,
)

from repolya.autogen.workflow import (
    create_rag_task_list_zh,
    search_faiss_openai,
)

import shutil
import json
import re
import os


def clean_filename(text, max_length=10):
    # 移除非法文件名字符（例如: \ / : * ? " < > |）
    _clean = re.sub(r'[\\/*?:"<>|]', '', text)
    # 替换操作系统敏感的字符
    _clean = _clean.replace(' ', '_')  # 替换空格为下划线
    # 取前 max_length 个字符作为文件名
    return _clean[:max_length]


def search_all(_query):
    _all = []
    _all.extend(google(_query, n=1))
    _all.extend(bing(_query, n=1))
    _all.extend(ddg(_query, n=1))
    return _all


def print_search_all(_all):
    _str = []
    for i in _all:
        _str.append(f"{i['link']}\n{i['title']}")
        # _str.append(f"{i['link']}\n{i['title']}\n{i['snippet']}")
    return "\n" + "\n".join(_str)


def task_with_context_template(_task, _context, template):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    prompt = PromptTemplate.from_template(template)
    with get_openai_callback() as cb:
        chain = prompt | llm | StrOutputParser()
        _ans = chain.invoke({"_task": _task, "_context": _context})
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
    return [_ans, _token_cost]


##### event context
yj_keyword = {
    "基本情况_1": "发生发展时间线",
    "基本情况_2": "灾害规模和强度",
    "处置过程_1": "实施应急响应和救援措施的时间线",
    "处置过程_2": "政府和非政府组织角色",#
    "军民协作_1": "军队参与救灾行动的时间线",
    "军民协作_2": "军队救援行动和协作细节",
    "法规依据_1": "军队协助救灾的法律依据",
    "法规依据_2": "应急救援军地联动机制",
    "影响评估_1": "灾害对经济和社会的影响",
    "影响评估_2": "受灾群体和地区的恢复进程",
    "反思启示_1": "灾害管理和应对的有效性评估",
    "反思启示_2": "灾害的经验教训和改进措施",
}


def generate_search_dict_for_event(_event: str) -> dict[str]:
    _event_name = clean_filename(_event, 20)
    _event_dir = str(AUTOGEN_JD / _event_name)
    _dict = {}
    logger_yj.info("generate_search_dict_for_event：开始")
    for i in yj_keyword.keys():
        _i = f"{_event} AND {yj_keyword[i]}"
        _dict[i] = _i
        logger_yj.info(_i)
    logger_yj.info("generate_search_dict_for_event：完成")
    return _dict


def generate_context_for_each_query(_query: str, _db_name: str, _clean_txt_dir: str):
    _context, _token_cost = "", "Tokens: 0 = (Prompt 0 + Completion 0) Cost: $0"
    _all = search_all(_query)
    logger_yj.info(print_search_all(_all))
    _all_link = [i['link'] for i in _all]
    _urls = list(set(_all_link))
    if not os.path.exists(_db_name):
        urls_to_faiss_OpenAI(_urls, _db_name, _clean_txt_dir)
    else:
        logger_yj.info(f"'{_db_name}'已存在，无需 urls_to_faiss_OpenAI")
    ### multi query
    _vdb = get_faiss_OpenAI(_db_name)
    _ask = _query.replace(' AND ', ' ')
    _key = _query.split(" AND ")[1]
    _context_fp = os.path.join(os.path.dirname(_db_name), "_context.txt")
    _ans_fp = os.path.join(os.path.dirname(_db_name), "_ans.txt")
    if not os.path.exists(_ans_fp):
        _ans, _step, _token_cost = qa_vdb_multi_query(_ask, _vdb, 'stuff')
        with open(_ans_fp, "w") as f:
            f.write(_ans)
        ##### _context
        _context = clean_txt(_ans)
        ### 去除 _context 中的 [1] [50.14] 等标记
        _context = re.sub(r'\[\d+\]', '', _context)
        _context = re.sub(r'\[\d+\.\d+\]', '', _context)
        with open(_context_fp, "w") as f:
            f.write(_context)
    else:
        with open(_ans_fp, "r") as f:
            _ans = f.read()
        _context = clean_txt(_ans)
        ### 去除 _context 中的 [1] [50.14] [81-82] 等标记
        _context = re.sub(r'\[\d+\]', '', _context)
        _context = re.sub(r'\[\d+\.\d+\]', '', _context)
        _context = re.sub(r'\[\d+\-\d+\]', '', _context)
        ### 去除 _context 中那些只包含空白（如空格、制表符等）的行
        _context = re.sub(r'^\s+\n$', '\n', _context, flags=re.MULTILINE)
        if '- ' in _context:
            _clean = []
            _li = _context.split('\n')
            for i in _li:
                if '- ' in i:
                    _clean.append(i)
            _context = '\n'.join(_clean)
        with open(_context_fp, "w") as f:
            f.write(_context)
    logger_yj.info(_ask)
    # logger_yj.info(_ans)
    # logger_yj.info(_matches)
    logger_yj.info(_context)
    logger_yj.info(_token_cost)
    return _context, _token_cost, _urls


def generate_context_for_each_query_textgen(_query: str, _db_name: str, _clean_txt_dir: str, _textgen_url: str):
    _context, _token_cost, _urls = "", "Tokens: 0 = (Prompt 0 + Completion 0) Cost: $0", ""
    if not os.path.exists(_db_name):
        _all = search_all(_query)
        logger_yj.info(print_search_all(_all))
        _all_link = [i['link'] for i in _all]
        _urls = list(set(_all_link))
        urls_to_faiss_HuggingFace(_urls, _db_name, _clean_txt_dir)
    else:
        logger_yj.info(f"'{_db_name}'已存在，无需 urls_to_faiss_HuggingFace")
    ### multi query
    _vdb = get_faiss_HuggingFace(_db_name)
    _ask = _query.replace(' AND ', ' ')
    _key = _query.split(" AND ")[1]
    _context_fp = os.path.join(os.path.dirname(_db_name), "_context.txt")
    _ans_fp = os.path.join(os.path.dirname(_db_name), "_ans.txt")
    if not os.path.exists(_ans_fp):
        _ans, _step, _token_cost = qa_vdb_multi_query_textgen(_ask, _vdb, 'stuff', _textgen_url)
        if _token_cost == "":
            _token_cost = "Tokens: 0 = (Prompt 0 + Completion 0) Cost: $0"
        with open(_ans_fp, "w") as f:
            f.write(_ans)
        ##### _context
        _context = clean_txt(_ans)
        ### 去除 _context 中的 [1] [50.14] 等标记
        _context = re.sub(r'\[\d+\]', '', _context)
        _context = re.sub(r'\[\d+\.\d+\]', '', _context)
        with open(_context_fp, "w") as f:
            f.write(_context)
    else:
        with open(_ans_fp, "r") as f:
            _ans = f.read()
        _context = clean_txt(_ans)
        ### 去除 _context 中的 [1] [50.14] 等标记
        _context = re.sub(r'\[\d+\]', '', _context)
        _context = re.sub(r'\[\d+\.\d+\]', '', _context)
        if '- ' in _context:
            _clean = []
            _li = _context.split('\n')
            for i in _li:
                if '- ' in i:
                    _clean.append(i)
            _context = '\n'.join(_clean)
        with open(_context_fp, "w") as f:
            f.write(_context)
    logger_yj.info(_ask)
    # logger_yj.info(_ans)
    # logger_yj.info(_matches)
    logger_yj.info(_context)
    logger_yj.info(_token_cost)
    return _context, _token_cost, _urls


def context_report(_event: str, _title: str, _context: dict, _urls: dict):
    _report = []
    _report.append(_title)
    _section = [
        "基本情况",
        "处置过程",
        "军民协作",
        "法规依据",
        "影响评估",
        "反思启示",
    ]
    for i in _section:
        _section = []
        _section.append(f"# {i}")
        for j in _context.keys():
            if i in j:
                _section.append(f"## [{yj_keyword[j]}]({_urls[j][0]})\n{_context[j]}")
        _report.append("\n\n".join(_section))
    # _report.append(f"# {_event}相关链接" + "\n\n" + "\n".join(_urls)
    return "\n\n\n".join(_report)


def generate_event_context(_event: str, _dict: dict[str]) -> dict[str]:
    _event_name = clean_filename(_event, 20)
    _event_dir = str(AUTOGEN_JD / _event_name)
    _context = {}
    if not os.path.exists(_event_dir):
        os.makedirs(_event_dir)
    logger_yj.info("generate_context_for_search_list：开始")
    _tc = []
    _urls = {}
    for i in _dict.keys():
        i_key = _dict[i].split(" AND ")[1]
        i_db_name = os.path.join(_event_dir, f"{i_key}/yj_rag_openai")
        i_clean_txt_dir = os.path.join(_event_dir, f"{i_key}/yj_rag_clean_txt")
        i_context, i_token_cost, i_urls = generate_context_for_each_query(_dict[i], i_db_name, i_clean_txt_dir)
        _context[i] = i_context
        _tc.append(i_token_cost)
        _urls[i] = i_urls
    _token_cost = calc_token_cost(_tc)
    logger_yj.info(_token_cost)
    logger_yj.info("generate_context_for_search_list：完成")
    _title = f"'{_event}'事件脉络梳理报告"
    _report_fp = os.path.join(_event_dir, f"{_title}.md")
    if not os.path.exists(_report_fp):
        _context_str = json.dumps(_context, ensure_ascii=False, indent=4)
        _report = context_report(_event, _context, _urls)
        with open(_report_fp, "w") as f:
            f.write(_report)
    else:
        with open(_report_fp, "r") as f:
            _report = f.read()
    return _report, _report_fp


def generate_event_context_textgen(_event: str, _dict: dict[str], _textgen_url: str) -> dict[str]:
    _event_name = clean_filename(_event, 20)
    _event_dir = str(AUTOGEN_JD / _event_name)
    _context = {}
    if not os.path.exists(_event_dir):
        os.makedirs(_event_dir)
    logger_yj.info("generate_context_for_search_list：开始")
    _tc = []
    _urls = {}
    for i in _dict.keys():
        i_key = _dict[i].split(" AND ")[1]
        i_db_name = os.path.join(_event_dir, f"{i_key}/yj_rag_hf")
        i_clean_txt_dir = os.path.join(_event_dir, f"{i_key}/yj_rag_clean_txt")
        i_context, i_token_cost, i_urls = generate_context_for_each_query_textgen(_dict[i], i_db_name, i_clean_txt_dir, _textgen_url)
        _context[i] = i_context
        _tc.append(i_token_cost)
        _urls[i] = i_urls
    _token_cost = calc_token_cost(_tc)
    logger_yj.info(_token_cost)
    logger_yj.info("generate_context_for_search_list：完成")
    _title = f"'{_event}'事件脉络梳理报告"
    _report_fp = os.path.join(_event_dir, f"{_title}.md")
    if not os.path.exists(_report_fp):
        _context_str = json.dumps(_context, ensure_ascii=False, indent=4)
        _report = context_report(_event, _title, _context, _urls)
        with open(_report_fp, "w") as f:
            f.write(_report)
    else:
        with open(_report_fp, "r") as f:
            _report = f.read()
    return _report, _report_fp


##### event plan
def generate_event_plan(_event: str, _context:str) -> str:
    _event_name = clean_filename(_event, 20)
    _event_dir = str(AUTOGEN_JD / _event_name)
    # _db_name = os.path.join(_event_dir, f"yj_rag_openai")
    logger_yj.info("generate_event_plan：开始")
    _plan = f"{_context}\n\n【plan】"
    logger_yj.info("generate_event_plan：完成")
    return _plan



def fetch_all_link(_all, _event_dir):
    _txt_fp = []
    _all_link = [i['link'] for i in _all]
    _all_title = [i['title'] for i in _all]
    # print(_all_link)
    loader = WebBaseLoader()
    _re = loader.scrape_all(_all_link)
    for i in range(len(_re)):
        _fn = clean_filename(_all_title[i])
        _fp = os.path.join(_event_dir, f"{_fn}.txt")
        with open(_fp, "w") as wf:
            # _txt = _re[i].get_text()
            _txt = _re[i].get_text()
            wf.write(_txt)
        logger_yj.info(f"{_all_link[i]} -> {_fn}.txt")
        _txt_fp.append(_fp)
    return _txt_fp


def handle_fetch(_event_dir, _db_name, _clean_txt_dir):
    logger_yj.info(f"generate faiss_openai：开始")
    dir_to_faiss_OpenAI(_event_dir, _db_name, _clean_txt_dir)
    logger_yj.info(f"generate faiss_openai：{_db_name}")
    logger_yj.info(f"generate faiss_openai：完成")



# def generate_vdb_for_search_query(_query: list[str], _event_name: str):
#     _event_dir = str(AUTOGEN_JD / _event_name)
#     if not os.path.exists(_event_dir):
#         os.makedirs(_event_dir)
#         _db_name = str(AUTOGEN_JD / _event_dir / f"yj_rag_openai")
#         _clean_txt_dir = str(AUTOGEN_JD / _event_dir / f"yj_rag_clean_txt")
#         logger_yj.info("generate_vdb_for_search_query：开始")
#         # ### search, fetch, vdb
#         # for i in _query:
#         #     i_all = search_all(i)
#         #     i_psa = print_search_all(i_all)
#         #     logger_yj.info(f"'{i}'")
#         #     logger_yj.info(i_psa)
#         #     fetch_all_link(i_all, _event_dir)
#         # handle_fetch(_event_dir, _db_name, _clean_txt_dir)
#         ### search, load, vdb
#         _all = []
#         for i in _query:
#             i_all = search_all(i)
#             _all.extend(i_all)
#         _psa = print_search_all(_all)
#         logger_yj.info(_psa)
#         _all_link = [i['link'] for i in _all]
#         _urls = list(set(_all_link))
#         urls_to_faiss(_urls, _db_name, _clean_txt_dir)
#         logger_yj.info("generate_vdb_for_search_query：完成")
#     else:
#         logger_yj.info(f"'{_event_name}'专题已存在，无需 generate_vdb_for_search_query")
#         # shutil.rmtree(_event_dir)
#         # os.makedirs(_event_dir)


# def ask_vdb_with_source(_ask, _vdb):
#     llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
#     chain = RetrievalQAWithSourcesChain.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=_vdb.as_retriever(),
#     )
#     with get_openai_callback() as cb:
#         _res = chain({"question": _ask}, return_only_outputs=True)
#         _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
#         logger_yj.info(_token_cost)
#     return _res['answer'], _res['sources'], _token_cost


# def generate_event_context(_event: str, _event_name: str) -> str:
#     _db_name = str(AUTOGEN_JD / _event_name / f"yj_rag_openai")
#     _ask_vdb = str(AUTOGEN_JD / _event_name / 'ask_vdb.txt')
#     if os.path.exists(_ask_vdb):
#         os.remove(_ask_vdb)
#     logger_yj.info("generate_event_context：开始")
#     _task = "请生成关于'{_context}'的详细信息查询问题列表（一个问题一行）。确保只列出具体的问题，而不包括任何章节标题或编号。"
#     _ans, _tc = task_with_context_template(_task, _event, _gec_ask)
#     # logger_yj.info(f"\n{_ans}")
#     _extracted = re.findall(r'\s+-\s+(.*)\n', _ans)
#     for i in _extracted:
#         logger_yj.info(i)
#     logger_yj.info(_tc)
#     ### ask_vdb
#     logger_yj.info("ask_vdb：开始")
#     _vdb = get_faiss_OpenAI(_db_name)
#     _qas = []
#     _tc = []
#     for i in _extracted:
#         _ask = i + "如果找不到确切答案，请回答'无'。用简洁中文回答。"
#         ### with source
#         # i_ans, i_source, i_token_cost = ask_vdb_with_source(_ask, _vdb)
#         # i_qas = f"Q: {i}\nA: {i_ans.strip()}\nSource: {i_source}"
#         ### multi query
#         # i_ans, i_step, i_token_cost = qa_vdb_multi_query(_ask, _vdb, 'stuff')
#         ### autogen
#         # _task_list, c_token_cost = create_rag_task_list_zh(i)
#         # _context, s_token_cost = search_faiss_openai(_task_list, _vdb)
#         # i_ans, i_token_cost = qa_with_context_as_go(_ask, _context)
#         ###
#         i_ans = clean_txt(i_ans)
#         i_ans = i_ans.strip()
#         if i_ans != '':
#             i_qas = f"Q: {i}\nA: {i_ans}"
#             logger_yj.info(i_qas)
#             _qas.append(i_qas)
#             with open(_ask_vdb, "w") as f:
#                 f.write("\n\n".join(_qas))
#         _tc.append(c_token_cost)
#         _tc.append(s_token_cost)
#         _tc.append(i_token_cost)
#     _token_cost = calc_token_cost(_tc)
#     logger_yj.info(_token_cost)
#     logger_yj.info("ask_vdb：完成")
#     ### 
#     _context = "【_context】"
#     logger_yj.info("generate_event_context：完成")
#     return _context


# def generate_search_query_for_event(_event: str, _event_name: str) -> list[str]:
#     _event_dir = str(AUTOGEN_JD / _event_name)
#     _query = []
#     if not os.path.exists(_event_dir):
#         logger_yj.info("generate_search_query_for_event：开始")
#         _task = "请生成关于'{_context}'的信息查询列表。确保只列出具体的查询语句，而不包括任何章节标题或编号。"
#         _ans, _tc = task_with_context_template(_task, _event, _gsqfe)
#         # logger_yj.info(f"\n{_ans}")
#         _extracted = re.findall(r'\s+-\s+(.*)\n', _ans)
#         for i in _extracted:
#             logger_yj.info(i)
#         logger_yj.info(_tc)
#         _query = _extracted
#         logger_yj.info("generate_search_query_for_event：完成")
#     else:
#         logger_yj.info(f"'{_event_name}'专题已存在，无需 generate_search_query_for_event")
#     return _query


# _gsqfe = """你的任务是使用搜索引擎（例如谷歌）来搜集关于特定自然灾害事件的全面信息。你需要寻找的信息包括事件的基本概况、处置过程、军民协作、法规（政策）依据、影响评估以及反思和启示。请按照以下指南生成详尽的查询：

# 1. 基本概况：
#    - [_具体应急事件名称_] 发生发展的时间线
#    - [_具体应急事件名称_] 灾害强度和规模

# 2. 处置过程：
#    - [_具体应急事件名称_] 实施的应急响应和救援措施的时间线
#    - [_具体应急事件名称_] 灾害响应物资和人员部署

# 3. 军民协作：
#    - [_具体应急事件名称_] 军队参与救灾行动的时间线
#    - [_具体应急事件名称_] 军队救援行动和协作细节

# 4. 法规（政策）依据：
#    - [具体应急事件名称] 军队协助救灾的法律依据
#    - [具体应急事件名称] 地方政府和军队救灾合作的政策框架
   
# 4. 影响评估：
#    - [_具体应急事件名称_] 灾害对经济和社会的影响
#    - [_具体应急事件名称_] 受灾群体和地区的恢复进程

# 5. 反思和启示：
#    - [_具体应急事件名称_] 灾害管理和应对的有效性评估
#    - [_具体应急事件名称_] 从灾害中学到的教训和改进措施

# 请确保替换[_具体应急事件名称_]为你正在研究的事件名称，以定位准确相关的资料。根据搜索结果的详尽程度，适时调整或细化你的查询关键词。
# [_具体应急事件名称_]: {_context}
# {_task}:
# """


# _gec_ask = """面对'{_context}'的严峻挑战，国家和地方应急管理部门需迅速掌握全面而详尽的信息，以便形成有效的应对策略。现在，请你根据以下阶段性信息需求，生成一系列的查询问题，以帮助我们搜集有关该事件全程的关键信息：

# 1. 灾害发生与初期响应：
#    - 该自然灾害确切的发生时间和地点是什么？
#    - 灾害的强度和规模？
#    - 灾害发生后，首批采取的应急响应措施包括哪些？

# 2. 应急反应与处置经过：
#    - 灾害发展扩散的过程是怎样的？
#    - 灾害发展扩散的时间线如何？
#    - 实施的应急响应和救援措施具体包括哪些？
#    - 实施应急响应和救援措施的时间线如何？
#    - 灾害的关键转折点有哪些？
#    - 在灾害应对中，涉及的关键人物和组织扮演了何种角色？
#    - 响应过程中依据了哪些政策法规来指导行动？
#    - 灾害对环境、公共健康、经济以及社会秩序产生了哪些深远影响？

# 3. 灾情控制与过渡性措施：
#    - 灾情得到控制的时间点及其标志性事件是什么？
#    - 控制灾情后采取了哪些关键措施和策略？
#    - 针对灾害受影响人群实施了哪些过渡性救助和支持？

# 4. 恢复重建与法规政策实施：
#    - 灾后恢复和重建工作的起始时间和阶段性目标是什么？
#    - 重建期间，制定或修改了哪些相关法律法规？
#    - 恢复和重建措施的长期社会评价和生态影响如何？

# 5. 整体评估与教训吸取：
#    - 如何系统评价本次自然灾害的处置效果和应急管理能力？
#    - 这次事件对未来灾害风险评估和预防计划有哪些启示？
#    - 为提高未来应急响应和灾害管理能力，提出哪些具体的建议和策略？

# 通过回答这些详细的问题，我们希望能够建立一个关于'{_context}'的全方位、深入的事件报告。请为每个信息点生成具体的查询问题，以便我们对知识库进行详细的事实搜索和数据分析。
# {_task}:
# """


# def do_multi_search(msg):
#     _agents = [
#     ]
#     _out = str(WORKSPACE_AUTOGEN / "organizer_output.txt")
#     def validate_results_func():
#         with open(_out, "r") as f:
#             content = f.read()
#         return bool(content)
#     _organizer = Organizer(
#         name="Search Team",
#         agents=_agents,
#         validate_results_func=validate_results_func,
#     )
#     _organizer_conversation_result = _organizer.broadcast_conversation(msg)
#     match _organizer_conversation_result:
#         case ConversationResult(success=True, cost=_cost, tokens=_tokens):
#             print(f"✅ Organizer.Broadcast was successful. Team: {_organizer.name}")
#             print(f"📊 Name: {_organizer.name} Cost: {_cost}, tokens: {_tokens}")
#             with open(_out, "r") as f:
#                 content = f.read()
#             return content
#         case _:
#             print(f"❌ Organizer.Broadcast failed. Team: {_organizer.name}")

