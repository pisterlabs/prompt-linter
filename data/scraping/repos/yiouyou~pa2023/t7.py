import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from pprint import pprint
from time import sleep


def query_csv_list(_csv_list, _query):
    from langchain.agents import create_csv_agent
    from langchain.agents.agent_types import AgentType
    from dotenv import load_dotenv
    load_dotenv()
    from langchain.chat_models import ChatOpenAI
    for i_list in _csv_list:
        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            i_list,
            verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        _re = agent.run(_query)
        print(f"{i_list}: {_re}")
        sleep(1)


def get_tables(_url):
    from pathlib import Path
    _pwd = Path(__file__).absolute()
    _module_path = _pwd.parent.parent
    print(_module_path)
    import sys
    sys.path.append(str(_module_path))
    from module.util import timestamp_now
    _ts = timestamp_now()
    r = requests.get(_url)
    soup = bs(r.text, 'html.parser')
    _t0 = soup.find('table')
    if _t0:
        _is_table = True
    else:
        _is_table = False
    i = 0
    _csv_list = []
    while _is_table:
        _p0 = _t0.find_previous_sibling('p').get_text()
        pprint(_p0)
        _d0 = pd.read_html(_t0.prettify())
        pprint(_d0)
        _csv = f"{_ts}_{i+1}.csv"
        _d0[0].to_csv(_csv, index = False)
        _csv_list.append(_csv)
        i += 1
        _t1 = _t0.find_next('table')
        if _t1:
            _is_table = True
            _t0 = _t1
        else:
            _is_table = False
    return _csv_list


def get_csv_list(_dir):
    _csv_list = []
    import os
    for (root, dirs, files) in os.walk(_dir, topdown=True):
        for name in files:
            _f = os.path.join(root, name)
            if '.csv' in _f:
                _csv_list.append(_f)
        for name in dirs:
            _f = os.path.join(root, name)
            if '.csv' in _f:
                _csv_list.append(_f)
    _csv_list = sorted(_csv_list)
    return _csv_list



if __name__ == "__main__":

    _url = 'https://learn.microsoft.com/en-us/azure/virtual-machines/disks-types'
    # _csv_list = get_tables(_url)
    # _csv_list = get_csv_list("./")
    # print(_csv_list)
    _csv_list = ["./1688285921_1.csv"]
    _query = "can Ultra Disk be used as OS Disk?"
    query_csv_list(_csv_list, _query)


