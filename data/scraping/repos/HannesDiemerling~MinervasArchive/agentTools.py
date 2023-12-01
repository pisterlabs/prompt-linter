from langchain.agents import Tool
from htmlTemplates import css, bot_template, user_template, disclaimer_text, box_template, user_img, bot_img
from typing import List
from langchain.agents import Tool
from streamlit.components.v1 import html
from agentFunctions import simple_report_search, report_summarizer, one_person_search, tearm_search

def create_tools():
    # define usable Tools for the Agent
    tools = [
        Tool(
            name = "TermSearch",
            func=tearm_search,
            description="use this tool if you are not sure about a term. Input the term"
        ),
        Tool(
            name = "SimpleReportSearch",
            func=simple_report_search,
            description="useful if you think that you need just a little information from the report to answer the User Question. Input a question what information you need and keywords, Suitable for a keywords-based search in a vector space"
        ),
        Tool(
            name = "ReportSummarizer",
            func = report_summarizer,
            description="useful if you think that you need a lot information from the report to answer the User Question. Input a question what information you need and keywords, Suitable for a keywords-based search in a vector space"
        ),
        Tool(
            name = "OnePersonSearch",
            func= one_person_search,
            description="useful if you think that you need personal information about a persons in the MPI to answer the User Question. Input a question with the name of the person you search for, Suitable for a keyword-based search in a vector space"
        )
    ]
    return tools
