# https://python.langchain.com/docs/integrations/tools/arxiv
# https://python.langchain.com/docs/integrations/providers/arxiv
# https://python.langchain.com/docs/integrations/document_loaders/arxiv
# %%
from langchain.document_loaders import ArxivLoader

# %%
#  https://arxiv.org/abs/2303.06094
# %%
docs = ArxivLoader(query="2303.06094", load_max_docs=1).load()
len(docs)
# %%
docs[0].metadata  # meta-information of the Document
# %%
docs[0].page_content[:40]
# %%
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
from langchain.agents import AgentType, initialize_agent, load_tools

# %%
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.0)
tools = load_tools(
    ["arxiv"],
)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
# %%
# 2303.06094
agent_chain.run(
    "What's the paper 2303.06094 about?",
)
# %%
