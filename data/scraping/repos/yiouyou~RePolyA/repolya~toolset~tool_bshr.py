from langchain.tools import tool
from langchain.tools import StructuredTool

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks import get_openai_callback

from repolya._const import WORKSPACE_RAG
from repolya._log import logger_toolset
from repolya.app.bshr.prompt import (
    SYS_BRAINSTORM,
    SYS_BRAINSTORM_ZH,
    SYS_HYPOTHESIS,
    SYS_HYPOTHESIS_ZH,
    SYS_SATISFICE,
    SYS_SATISFICE_ZH,
    SYS_REFINE,
    SYS_REFINE_ZH,
)
from repolya.rag.qa_chain import qa_vdb_multi_query
from repolya.rag.vdb_faiss import get_faiss_OpenAI
from repolya.toolset.util import calc_token_cost

# from halo import Halo
import concurrent.futures as cf
import requests
import json
import re


def _chain(_sys: str, _human: str):
    _re, _token_cost = "", ""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _sys),
            ("human", "{text}"),
        ]
    )
    model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    runnable = (
        {"text": RunnablePassthrough()}
        | prompt 
        | model 
        | StrOutputParser()
    )
    with get_openai_callback() as cb:
        _re = runnable.invoke({"text": _human})
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
    return _re, _token_cost


##### brainstorm
def search_wiki(query: str) -> (str, str):
    # spinner = Halo(text='Information Foraging...', spinner='dots')
    # spinner.start()
    url = 'https://en.wikipedia.org/w/api.php'
    search_params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json'
    }
    response = requests.get(url, params=search_params)
    data = response.json()
    title = data['query']['search'][0]['title']
    content_params = {
        'action': 'query',
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
        'titles': title,
        'format': 'json'
    }
    response = requests.get(url, params=content_params)
    data = response.json()
    page_id = list(data['query']['pages'].keys())[0]
    content = data['query']['pages'][page_id]['extract']
    url = f"https://en.wikipedia.org/?curid={page_id}"
    # spinner.stop()
    return content, url

def brainstorm_wiki(_query: str, _notes: str, _queries: str):
    _tc = []
    _sys = SYS_BRAINSTORM
    _spr = SYS_REFINE
    _human = f"""
# USER QUERY
{_query}


# NOTES
{_notes}


# PREVIOUS QUERIES
{_queries}
"""
    _re, _token_cost = _chain(_sys, _human)
    _tc.append(_token_cost)
    logger_toolset.info(f"new questions: {_re}")
    _questions = json.loads(_re)
    for _q in _questions:
        content, url = search_wiki(_q)
        compressed_content, _spc_tc = _chain(_spr, content)
        _tc.append(_spc_tc)
        _notes = f"{_notes}\n\nURL: {url}\nNOTE: {compressed_content}"
        logger_toolset.info(_q)
        logger_toolset.info(url)
        # logger_toolset.info(content)
        logger_toolset.info(compressed_content)
        _queries = f"""
{_queries}

QUESTION: {_q}

"""
    return _queries, _notes, calc_token_cost(_tc)

def brainstorm_wiki_zh(_query: str, _notes: str, _queries: str):
    _tc = []
    _sys = SYS_BRAINSTORM_ZH
    _spr = SYS_REFINE_ZH
    _human = f"""
# USER QUERY
{_query}


# NOTES
{_notes}


# PREVIOUS QUERIES
{_queries}
"""
    _re, _token_cost = _chain(_sys, _human)
    _tc.append(_token_cost)
    logger_toolset.info(f"new questions: {_re}")
    _questions = json.loads(_re)
    for _q in _questions:
        content, url = search_wiki(_q)
        compressed_content, _spc_tc = _chain(_spr, content)
        _tc.append(_spc_tc)
        _notes = f"{_notes}\n\nURL: {url}\nNOTE: {compressed_content}"
        logger_toolset.info(_q)
        logger_toolset.info(url)
        # logger_toolset.info(content)
        logger_toolset.info(compressed_content)
        _queries = f"""
{_queries}

QUESTION: {_q}

"""
    return _queries, _notes, calc_token_cost(_tc)


def search_vdb(_query, _vdb):
    _ans, _step, _token_cost = qa_vdb_multi_query(_query, _vdb, 'stuff')
    return _ans, _token_cost

def brainstorm_vdb(_query, _notes, _queries, _vdb):
    _tc = []
    _sys = SYS_BRAINSTORM
    _spr = SYS_REFINE
    _human = f"""
# USER QUERY
{_query}


# NOTES
{_notes}


# PREVIOUS QUERIES
{_queries}
"""
    _re, _token_cost = _chain(_sys, _human)
    _tc.append(_token_cost)
    logger_toolset.info(f"new questions: {_re}")
    _questions = json.loads(_re)
    for _q in _questions:
        content, _vdb_tc = search_vdb(_q, _vdb)
        _tc.append(_vdb_tc)
        compressed_content, _spc_tc = _chain(_spr, content)
        _tc.append(_spc_tc)
        _notes = f"{_notes}\n\nNOTE: {compressed_content}"
        logger_toolset.info(_q)
        # logger_toolset.info(content)
        logger_toolset.info(compressed_content)
        _queries = f"""
{_queries}

QUESTION: {_q}

"""
    return _queries, _notes, calc_token_cost(_tc)

def brainstorm_vdb_zh(_query, _notes, _queries, _vdb):
    _tc = []
    _sys = SYS_BRAINSTORM_ZH
    _spr = SYS_REFINE_ZH
    _human = f"""
# USER QUERY
{_query}


# NOTES
{_notes}


# PREVIOUS QUERIES
{_queries}
"""
    _re, _token_cost = _chain(_sys, _human)
    _tc.append(_token_cost)
    logger_toolset.info(f"new questions: {_re}")
    _questions = json.loads(_re)
    for _q in _questions:
        content, _vdb_tc = search_vdb(_q, _vdb)
        _tc.append(_vdb_tc)
        compressed_content, _spc_tc = _chain(_spr, content)
        _tc.append(_spc_tc)
        _notes = f"{_notes}\n\nNOTE: {compressed_content}"
        # logger_toolset.info(_q)
        # logger_toolset.info(content)
        logger_toolset.info(compressed_content)
        _queries = f"""
{_queries}

QUESTION: {_q}

"""
    return _queries, _notes, calc_token_cost(_tc)


##### hypothesize
def hypothesize(_query: str, _notes: str, _hypotheses: str):
    _sys = SYS_HYPOTHESIS
    _human = f"""
# USER QUERY
{_query}


# NOTES
{_notes}


# PREVIOUS HYPOTHISES
{_hypotheses}
"""
    _re, _token_cost = _chain(_sys, _human)
    # logger_toolset.info(f"new hypothesis: '{_re}'")
    return _re, _token_cost

def hypothesize_zh(_query: str, _notes: str, _hypotheses: str):
    _sys = SYS_HYPOTHESIS_ZH
    _human = f"""
# USER QUERY
{_query}


# NOTES
{_notes}


# PREVIOUS HYPOTHISES
{_hypotheses}
"""
    _re, _token_cost = _chain(_sys, _human)
    # logger_toolset.info(f"new hypothesis: '{_re}'")
    return _re, _token_cost


##### satisfice
def satisfice(_query: str, _notes: str, _queries: str, _hypothesis: str):
    _sys = SYS_SATISFICE
    _human = f"""# USER QUERY
{_query}


# NOTES
{_notes}


# QUERIES AND ANSWERS
{_queries}


# FINAL HYPOTHESIS
{_hypothesis}

"""
    _re, _token_cost = _chain(_sys, _human)
    _feedback = json.loads(_re)
    return _feedback["satisficed"], _feedback["feedback"], _token_cost

def satisfice_zh(_query: str, _notes: str, _queries: str, _hypothesis: str):
    _sys = SYS_SATISFICE_ZH
    _human = f"""# USER QUERY
{_query}


# NOTES
{_notes}


# QUERIES AND ANSWERS
{_queries}


# FINAL HYPOTHESIS
{_hypothesis}

"""
    _re, _token_cost = _chain(_sys, _human)
    _feedback = json.loads(_re)
    return _feedback["satisficed"], _feedback["feedback"], _token_cost


##### refine
def refine(_notes: str):
    _sys = SYS_REFINE
    _human = _notes
    _re, _token_cost = _chain(_sys, _human)
    return _re, _token_cost

def refine_zh(_notes: str):
    _sys = SYS_REFINE_ZH
    _human = _notes
    _re, _token_cost = _chain(_sys, _human)
    return _re, _token_cost


##### bshr + wiki
def bshr_wiki(_query: str):
    logger_toolset.info(f"query: '{_query}'")
    _tc = []
    notes = ""
    queries = ""
    iteration = 0
    max_iterations = 3
    hypotheses_feedback = "# FEEDBACK ON HYPOTHESES\n"
    while True:
        iteration += 1
        logger_toolset.info(f"iteration ({iteration}) started")
        new_queries, notes, _token_cost = brainstorm_wiki(
            _query=_query,
            _notes=notes, 
            _queries=queries,
        )
        queries += new_queries
        _tc.append(_token_cost)
        new_hypothesis, _token_cost = hypothesize(
            _query=_query,
            _notes=notes,
            _hypotheses=hypotheses_feedback,
        )
        _tc.append(_token_cost)
        satisficed, feedback, _token_cost = satisfice(
            _query=_query,
            _notes=notes,
            _queries=queries,
            _hypothesis=new_hypothesis,
        )
        _tc.append(_token_cost)
        hypotheses_feedback = f"""
{hypotheses_feedback}

## HYPOTHESIS
{new_hypothesis}

## FEEDBACK
{feedback}
"""
        logger_toolset.info(f"new_hypothesis: '{new_hypothesis}'")
        logger_toolset.info(f"satisficed: '{satisficed}'")
        logger_toolset.info(f"feedback: '{feedback}'")
        if satisficed or max_iterations <= iteration:
            logger_toolset.info(f"reached max iterations: {max_iterations <= iteration}")
            break
        notes, _token_cost = refine(notes)
        _tc.append(_token_cost)
        logger_toolset.info(f"iteration ({iteration}) completed")
    _re = new_hypothesis.split("\n\n")[-1]
    return _re, calc_token_cost(_tc)

def tool_bshr_wiki():
    tool = StructuredTool.from_function(
        bshr_wiki,
        name="BSHR with wikipedia (EN)",
        description="BSHR (Brainstorm, Hypothesize, Satisfice, Refine), information foraging with wikipedia.",
        verbose=True,
    )
    return tool


##### bshr + vdb
def bshr_vdb(_query, _db_name):
    _vdb = get_faiss_OpenAI(_db_name)
    logger_toolset.info(f"query: '{_query}'")
    _tc = []
    notes = ""
    queries = ""
    iteration = 0
    max_iterations = 1
    hypotheses_feedback = "# 对假设的反馈\n"
    while True:
        iteration += 1
        logger_toolset.info(f"iteration ({iteration}) started")
        if iteration == 1:
            logger_toolset.info("+[vdb]")
            new_queries, notes, _token_cost = brainstorm_vdb_zh(
                _query=_query,
                _notes=notes, 
                _queries=queries,
                _vdb=_vdb,
            )
        queries += new_queries
        _tc.append(_token_cost)
        new_hypothesis, _token_cost = hypothesize_zh(
            _query=_query,
            _notes=notes,
            _hypotheses=hypotheses_feedback,
        )
        _tc.append(_token_cost)
        satisficed, feedback, _token_cost = satisfice_zh(
            _query=_query,
            _notes=notes,
            _queries=queries,
            _hypothesis=new_hypothesis,
        )
        _tc.append(_token_cost)
        hypotheses_feedback = f"""
{hypotheses_feedback}

## HYPOTHESIS
{new_hypothesis}

## FEEDBACK
{feedback}
"""
        logger_toolset.info(f"new_hypothesis: '{new_hypothesis}'")
        logger_toolset.info(f"satisficed: '{satisficed}'")
        logger_toolset.info(f"feedback: '{feedback}'")
        if satisficed or max_iterations <= iteration:
            logger_toolset.info(f"reached max iterations: {max_iterations <= iteration}")
            break
        notes, _token_cost = refine_zh(notes)
        _tc.append(_token_cost)
        logger_toolset.info(f"iteration ({iteration}) completed")
    _re = new_hypothesis
    return _re, calc_token_cost(_tc)

