from langchain.agents.tools import Tool
from llms.azure_llms import create_llm
from tools.prebuilt_tools import python_tool, wikipedia_tool, duckduckgo_tool, human_tool
#from tools.prebuilt_tools import shell_tool
from tools.qa_tools import qa_retrieval_tool

from tools.borealis_tools import borealis_tool
from tools.opencti_tools import openCTI_tool
from tools.shodan_tools import shodan_ip_lookup_tool
from tools.ipapi_tools import ipapi_tool
from tools.kendra.tool import kendra_retrieval_tool
from tools.abuseIPDB_tools import abuseIPDB_check_IP


tool_llm = create_llm()

base_tools=[
    python_tool,
    wikipedia_tool,
    duckduckgo_tool,
    qa_retrieval_tool,
    borealis_tool,
    openCTI_tool,
    shodan_ip_lookup_tool,
    ipapi_tool,
    abuseIPDB_check_IP
    # kendra_retrieval_tool
    # ip_report_tool
]

qa_tools=[
    kendra_retrieval_tool
]