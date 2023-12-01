from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import LLM
import itertools
import streamlit as st

from demo.evaluators.cold_email_binary_evaluator.cold_email_binary_eval_chain import SDREvalChain

@st.cache_data
def evaluate(customers: List[str], emails: List[str], _llm: LLM = ChatOpenAI(temperature=0)):
    """
    Generate inferences
    @param inputs: text to generate inferences from
    @param _llm: text to generate inferences from
    @return: inference set as JSON list
    """
    chain = SDREvalChain.from_llm(_llm)
    infs = chain.evaluate(customers, emails)
    return infs