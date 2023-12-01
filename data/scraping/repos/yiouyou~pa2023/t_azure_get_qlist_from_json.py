def writeF(_dir, _fn, _txt):
    import os
    wfn = os.path.join(_dir, _fn)
    with open(wfn, 'w', encoding='utf-8') as wf:
        wf.write(_txt)


def readF(_dir, _fn):
    import os
    rfn = os.path.join(_dir, _fn)
    with open(rfn, 'r', encoding='utf-8') as rf:
        return rf.read()


def parse_to_item(_ans):
    _item = []
    import re
    _li = _ans.split("\n")
    for i in _li:
        i = i.strip()
        _m = ''
        if ':' in i:
            m1 = re.search(r"^\d+\. (.+)\: ", i, re.DOTALL)
            if m1 is not None:
                _m = m1.group(1)
        elif ']' in i:
            m2 = re.search(r"^\d+\. \[(.+)\]", i, re.DOTALL)
            if m2 is not None:
                _m = m2.group(1)
        else:
            m3 = re.search(r"^\d+\. (.+)", i, re.DOTALL)
            if m3 is not None:
                _m = m3.group(1)
        if _m:
            if ' - ' in _m:
                _ms = _m.split(' - ')
                _item.append(_ms[-1].strip())
            else:
                _item.append(_m.strip())
    if not _item:
        print("\n\n###################################################")
        print(_ans)
        exit()
    # print(_item)
    return _item

# _ans = """1. SQL Database - Single database
# 2. SQL Database - Elastic pool"""
# _ans="""1. vCore-based purchasing model: This model allows you to choose the number of vCores, the amount of memory, and the amount and speed of storage. It offers more flexibility and control over resource allocation. You can also use Azure Hybrid Benefit for SQL Server to save costs by leveraging your existing SQL Server licenses.

# 2. DTU-based purchasing model: This model offers a blend of compute, memory, and I/O resources in three service tiers (Basic, Standard, and Premium) to support different database workloads. Each tier has different compute sizes and allows you to add additional storage resources."""
# _ans = """1. [SQL Managed Instance - Single instance](/en-us/pricing/details/azure-sql-managed-instance/single/)
# 2. [SQL Managed Instance - Instance pool](/en-us/pricing/details/azure-sql-managed-instance/pools/)"""
# print(parse_to_item(_ans))
# exit()


def qa_and_parse_to_item(_q, _dj, _service):
    from pathlib import Path
    _pwd = Path(__file__).absolute()
    _module_path = _pwd.parent.parent
    sys.path.append(str(_module_path))
    from module.query_vdb import qa_faiss_multi_query
    from dotenv import load_dotenv
    load_dotenv()
    _vdb = _dj[_service]['vdb']
    _ans, _step = qa_faiss_multi_query(_q, _vdb)
    # print(_ans)
    _item = parse_to_item(_ans)
    # print(_item)
    return _item


def what_and_stepbystep_explanation(_topic, _dj, _service):
    _list = []
    _q0 = f"How many {_topic} does {_service} have, and what are they? Please output in Numbered List."
    _list.append(_q0)
    _item = qa_and_parse_to_item(_q0, _dj, _service)
    if _item:
        for i in _item:
            _qi = ""
            if _topic in ["performance metrics", "pricing options"]:
                _qi = f"Can you provide a step-by-step explanation of how to use the {i} as one of {_topic}?"
            elif _topic in ["cost drivers"]:
                _qi = f"What are the best practices for choosing suitable {i} based on usage to save cost?"
            if _qi:
                _list.append(_qi)
            else:
                print("ERROR: wrong topic")
    return _list


def comparison_between_itmes(_c1, _c2, _dj, _service):
    _list = []
    _g = _dj[_service]['qlist'][_c1]['key_concept'][_c2]
    for i in _g:
        i_q0 = f"What are the unique features of '{i}' in {_c2}?"
        i_q1 = f"What are the limitations of '{i}' in {_c2}?"
        i_q2 = f"When should I choose '{i}' in {_c2}?"
        _list.extend([i_q0, i_q1, i_q2])
    for i in range(len(_g)):
        for j in range(i+1, len(_g)):
            print(_g[i], _g[j])
            _gi = f"'{_g[i]}' in {_c2}"
            _gj = f"'{_g[j]}' in {_c2}"
            j_q0 = f"What's the difference between {_gi} and {_gj}?"
            j_q1 = f"When should I choose {_gi} over {_gj}?"
            j_q2 = f"When should I choose {_gj} over {_gi}?"
            _list.extend([j_q0, j_q1, j_q2])
    # print(_list)
    return _list


def qlist_from_json(_dj, _service):
    _d = _dj[_service]['qlist']
    # print(_d)
    _list = []
    for i in _d.keys():
        i_list = what_and_stepbystep_explanation(i, _dj, _service)
        _list.extend(i_list)
        if 'key_concept' in _d[i]:
            _d_kc = _d[i]['key_concept']
            for j in _d_kc.keys():
              # print(i, j)
              if _d_kc[j]:
                  print(j)
                  _compare_list = comparison_between_itmes(i, j, _dj, _service)
                  _list.extend(_compare_list)
              else:
                  print()
                  _qj = f"Can you provide a step-by-step explanation of how the {j} affect the cost of {_service}?"
                  _list.extend([_qj])
        # else:
        #     print(i)
    _qlist = sorted(list(set(_list)))
    return _qlist


def get_ans_from_qlist(_qlist, _dir, _dj, _service):
    _qlist_str = "\n".join(_qlist)
    writeF(_dir, '_qlist', _qlist_str)
    from pathlib import Path
    _pwd = Path(__file__).absolute()
    _module_path = _pwd.parent.parent
    sys.path.append(str(_module_path))
    from module.query_vdb import qa_faiss_multi_query
    from dotenv import load_dotenv
    load_dotenv()
    import time
    _vdb = _dj[_service]['vdb']
    _ans = []
    for i in _qlist:
        _q = f"{i} Please output in concise English."
        i_ans, i_step = qa_faiss_multi_query(_q, _vdb)
        writeF(_dir, '_ans_/_ans_'+i.replace("?", ""), i_ans)
        writeF(_dir, '_step_/_step_'+i.replace("?", ""), i_step)
        time.sleep(4)
        _ans.append(i_ans)
    _ans_str = ""
    for i in range(len(_qlist)):
        _ans_str += f"## {_qlist[i]}\n\n" + f"{_ans[i]}\n\n"
    writeF(_dir, '_ans', _ans_str)


def _chatmodel(_ans_str, _dir, _service):
    _info = _ans_str
    _out_rule = '_rule'
    _out_step = '_rule_step'
    from dotenv import load_dotenv
    load_dotenv()
    from langchain.callbacks import get_openai_callback
    from langchain.prompts.prompt import PromptTemplate
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.chat_models import JinaChat
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    sys_template = (
        " You are a cost optimization expert, providing cost optimization suggestions for Azure cloud service customers. In order to achieve this goal, it is necessary to first construct a list of cost optimization rules, listing what can and cannot be done in various situations; then write python code according to the cost optimization rules, which is related to inputting the usage status of customer cloud services When using data, all feasible optimization measures can be directly calculated and recommended with priority of cost and safety."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_template = \
"""
Giving the following information:
--------------------
{info}
--------------------
Given a comprehensive FAQ document on {service}, please extract all information related to cost optimization strategies, usage-based billing, fixed costs services, cost-saving features, best practices for choosing suitable services based on usage. Also, provide detailed information on monitoring usage metrics, understanding billing details, tracking consumption trends, and how these can contribute to cost savings. Furthermore, include instances where changing service plans or tiers might lead to cost savings. Compile all this information into a comprehensive cost optimization rule book for writing Python programs for Azure users.
"""
# What are necessary non-duplicative rules that you can extract to optimize the usage of {service}? Remember to cover as many details as possible. Only output non-duplicative, nothing else:
# """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    rule_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    import os, re
    with get_openai_callback() as cb:
        # llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0)
        chain = LLMChain(llm=llm, prompt=rule_prompt)
        _re = chain.run(info=_info, service=_service)
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        _rule = _re.strip().split("\n")
        _step4 = f"{_token_cost}\n\n" + "="*20+" prompt "+"="*20+"\n" + rule_prompt.format(info=_info, service=_service) + "="*20+" prompt "+"="*20+"\n" + f"extracted rules:\n\n" + "\n".join(_rule)
    writeF(_dir, _out_rule, "\n".join(_rule))
    writeF(_dir, _out_step, _step4)


def extract_rules(_ans_f, _dir, _service):
    with open(_ans_f, 'r', encoding='utf-8') as rf:
        _ans_str = rf.read()
    _chatmodel(_ans_str, _dir, _service)



if __name__ == "__main__":

    import os, sys, json
    ##### get json
    _json = sys.argv[1]
    _service = sys.argv[2]
    # _service = "azure sql managed instance"
    ##### get qlist
    with open(_json, 'r', encoding='utf-8') as jf:
        _dj = json.loads(jf.read())
    _qlist = qlist_from_json(_dj, _service)
    for i in _qlist:
        print(i)
    ##### get ans
    from module.util import timestamp_now
    _ts = timestamp_now()
    _service_str = '_'.join(_service.split(' '))
    _dir = f"tmp_{_service_str}_{_ts}"
    _dir_ans_ = f"{_dir}/_ans_"
    _dir_step_ = f"{_dir}/_step_"
    # if not os.path.exists(_dir):
    #     os.makedirs(_dir)
    # if not os.path.exists(_dir_ans_):
    #     os.makedirs(_dir_ans_)
    # if not os.path.exists(_dir_step_):
    #     os.makedirs(_dir_step_)
    # get_ans_from_qlist(_qlist, _dir, _dj, _service)
    ##### get rules
    # _ans_f = os.path.join(_dir, '_ans')
    # extract_rules(_ans_f, _dir, _service)
    # python t_azure_get_qlist_from_json.py t_azure_get_qlist_from_json.json "azure managed disk"
    # python t_azure_get_qlist_from_json.py t_azure_get_qlist_from_json.json "azure sql managed instance"
    # python t_azure_get_qlist_from_json.py t_azure_get_qlist_from_json.json "azure sql database"
    # python t_azure_get_qlist_from_json.py t_azure_get_qlist_from_json.json "azure static web apps"


    # _ans_f = os.path.join(sys.argv[1], '_ans')
    # extract_rules(_ans_f, sys.argv[1], sys.argv[2])
    # python t_azure_get_qlist_from_json.py tmp_azure_managed_disk_1690514432 "azure managed disk"

