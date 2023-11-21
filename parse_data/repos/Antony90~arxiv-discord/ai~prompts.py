# ============= #
# AGENT PROMPTS #
# ============= #

from langchain import PromptTemplate

AGENT_PROMPT = \
"""You are arXiv Chat, an expert research assistant with access to a PDF papers.
You are also a discord bot whose goal is to make the process of literature exploration more efficient, facilitating discussions across multiple papers, as well as with peers.
Human messages are formatted <discord username>: <message>. You must address the discord user directly.

Use markdown syntax whenever appopriate: markdown headers, bullet point lists etc. but never use markdown links. Prefer bullet points over numbered lists.
Never output a paper abs/pdf link, only paper ID.

IMPORTANT:
At the end of every response, always tell the user what they can do next by suggesting functions they can make you call.
Always confirm with the user before executing a function, ask them whether it should be used with the arguments you've thought of.
Use functions only if explicitly asked by the user, they are expensive to use. Direct the user elsewhere if your functions are not appropriate.
The output of all functions must be kept unchanged when used in a response."""
# You must always respond succinctly, in as little words as possible; do not without decorate your responses.
# Only use tools if strictly necessary or are definitely related to a loaded paper.

PAPERS_PROMPT = \
"""These are papers which have been mentioned in your conversation. Use these paper IDs in tools.
If you are unsure which paper ID should be used in a tool, always ask for clarification.
{papers}
"""



# ============ #
# TOOL PROMPTS #
# ============ #

SEARCH_TOOL = \
"""Search arXiv and get a list of relevant papers (title and ID). You may rephrase a question to be a better search query. Always as the user before using this tool."""
# Assume the user wants you to search arXiv if given a vague set of terms or if asked to find/search.

# paper_title cannot be easily substituted at runtime, so generate the prompt with it fixed
MULTI_QUERY_PROMPT = lambda title: PromptTemplate(
template=f"""You are an expert reserach assistant with access to arXiv papers.
Your task is to generate 3 different versions of the given user 
question to retrieve relevant documents from a vector database for a paper titled {title}.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations 
of distance-based similarity search. Provide these alternative questions seperated by newlines. 
Original question: {{question}}""",
input_variables=["question"]
)



# SUMMARIZATION

MAP_PROMPT = PromptTemplate(input_variables=["text"], template=\
"""Summarize this text from an academic paper. Extract any key points with reasoning:

"{text}"

Summary:
""")


REDUCE_KEYPOINTS_PROMPT = PromptTemplate(input_variables=["text"], template=\
"""Write a summary collated from this collection of key points extracted from an academic paper.
The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
The summary should be structured in markdown bulleted lists (with optional sub-bulletpoints) following the headings Core Argument, Evidence, and Conclusions but this can be adapted.
Key points:
{text}

Summary:
""")


REDUCE_LAYMANS_PROMPT = PromptTemplate(input_variables=["text"], template=\
"""Write a laymans summary of the key points from a paper, focussing on objective fact:


"{text}"


Summary:""")
REDUCE_COMPREHENSIVE_PROMPT = PromptTemplate(input_variables=["text"], template=\
"""Write a concise summary of the following paper, focussing on objective fact:


"{text}"


Summary:""")


ABSTRACT_SUMMARY_PROMPT = PromptTemplate(input_variables=["title", "abstract"], template=\
"""Given the following abstract from the paper {title}, write a short bullet point summary, possibly highlighting
key findings, contributions, potential implications/applications, impact and methodology in heading sections.

ABSTRACT:
{abstract}

SUMMARY:""")


ABSTRACT_QS_PROMPT = PromptTemplate(input_variables=["title", "abstract"], template=\
"""Given the following abstract from the paper {title}, generate up to 5 concise questions to jump start an
in-depth discussion between expert researchers. Ensure they prompt discussion on: the findings put forward, the core argument, the key take-aways/conclusions.

ABSTRACT:
{abstract}

QUESTIONS:""")