from ._tools import search_google_serp
from ._tools import search_serp
from ._tools import search_goolge
from ._tools import search_duck
from ._tools import docstore_wiki
from ._tools import wiki_query
from ._tools import wolfram_alpha
from ._tools import llm_math
from ._tools import retriev_azure
from ._tools import pubmed_query
from ._tools import youtube_search
from ._tools import human_input
from ._tools import weather_map
from ._tools import cp_file, del_file, search_file, mv_file, read_file, write_file, list_dir
from langchain.agents import Tool


all_tools = [
    Tool(
        name="Intermediate Answer",
        description="useful for when you need to ask with search",
        func=search_google_serp.run
    ),
    Tool(
        name="Search google serp",
        description="useful for when you need to answer questions about current events or the current state of the world",
        func=search_google_serp.run,
    ),
    Tool(
        name = "Search serp",
        description="useful for when you need to answer questions about current events or the current state of the world",
        func=search_serp.run
    ),
    Tool(
        name="Search google",
        description="useful for when you need to answer questions about current events or the current state of the world",
        func=search_goolge.run
    ),
    Tool(
        name="Search",
        description="useful for when you need to ask with search",
        func=docstore_wiki.search
    ),
    Tool(
        name="Lookup",
        description="useful for when you need to ask with lookup",
        func=docstore_wiki.lookup
    ),
    Tool(
        name="Calculator",
        description="useful for when you need to answer questions about math",
        func=llm_math.run
    ),
    Tool(
        name="QA for Azure",
        description="useful for when you need to query information about Azure virtual machine and disks",
        func=retriev_azure.run
    )
]


tools_selfask_googleserp = [all_tools[0]]
tools_selfask_azure = [
    Tool(
        name="Intermediate Answer",
        description="useful for when you need to ask with search",
        func=retriev_azure.run
    )
]

tools_googleserp = [all_tools[1]]
tools_serp = [all_tools[2]]
tools_google = [all_tools[3]]

tools_react_docstore_wiki = [all_tools[4], all_tools[5]]
tools_react_docstore_azure_googleserp = [
    Tool(
        name="Search",
        description="useful for when you need to ask with search",
        func=retriev_azure.run,
    ),
    Tool(
        name="Lookup",
        description="useful for when you need to ask with lookup",
        func=search_google_serp.run,
    )
]

tools_faiss_azure = [all_tools[7]]
tools_faiss_azure_math = [all_tools[7], all_tools[6]]

tools_faiss_azure_googleserp = [all_tools[7], all_tools[1]]
tools_faiss_azure_googleserp_math = [all_tools[7], all_tools[1], all_tools[6]]

