#!/usr/bin/env python
# coding: utf-8
import streamlit as st

st.set_page_config(page_title="Genesis Demo", page_icon=":robot:")


import os
import getpass
import pandas as pd
import re

# Use environment variable to set OpenAI API key if already set (just leave this code commented out if already set)
# secret_key = getpass.getpass("Enter OpenAI secret key: ")
# os.environ["OPENAI_API_KEY"] = secret_key


# Or use Vicuna via Shale Protocol (free tier)

# os.environ['OPENAI_API_BASE'] = "https://shale.live/v1"
# shale_api = getpass.getpass('Enter Shale API key: ')
# os.environ['OPENAI_API_KEY'] = shale_api


# <a href="https://colab.research.google.com/github/ryderwishart/biblical-machine-learning/blob/main/gpt-inferences/greek-hebrew-tsv-qa-agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Question answer over MACULA Greek and Hebrew

# In[1]:

# !pip install tabulate pandas langchain

# ## Set up MACULA dataframes
verse_df = pd.read_csv("preprocessed-macula-dataframes/verse_df.csv")
mg = pd.read_csv("preprocessed-macula-dataframes/mg.csv")
# mh = pd.read_csv("preprocessed-macula-dataframes/mh.csv")

# ## Set up QA agent

# # Expand functionality for more tools using DB lookups

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

# bible_persist_directory = '/Users/ryderwishart/genesis/databases/berean-bible-database'
bible_persist_directory = "/Users/ryderwishart/genesis/databases/berean-bible-database"
bible_chroma = Chroma(
    "berean-bible", embeddings, persist_directory=bible_persist_directory
)
# print(bible_chroma.search("jesus speaks to peter", search_type="similarity", k=1))

encyclopedic_persist_directory = "/Users/ryderwishart/biblical-machine-learning/gpt-inferences/databases/encyclopedic"
encyclopedic_chroma = Chroma(
    persist_directory=encyclopedic_persist_directory,
    embedding_function=embeddings,
    collection_name="encyclopedic",
)
# print(
#     encyclopedic_chroma.similarity_search_with_score(
#         "What is a sarcophagus?", search_type="similarity", k=1
#     )
# )

theology_persist_directory = (
    "/Users/ryderwishart/biblical-machine-learning/gpt-inferences/databases/theology"
)
theology_chroma = Chroma(
    "theology", embeddings, persist_directory=theology_persist_directory
)
# print(theology_chroma.search("jesus speaks to peter", search_type="similarity", k=1))

# # persist_directory = '/Users/ryderwishart/genesis/databases/itemized-prose-contexts copy' # NOTE: Itemized prose contexts are in this db
# persist_directory = '/Users/ryderwishart/genesis/databases/prose-contexts' # NOTE: Full prose contexts are in this db
# context_chroma = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name="prosaic_contexts_itemized")
# print(context_chroma.similarity_search_with_score('jesus (s) speaks (v) to peter (o)', search_type='similarity', k=1))

persist_directory = (
    "/Users/ryderwishart/genesis/databases/prose-contexts-shorter-itemized"
)
context_chroma = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="prosaic_contexts_shorter_itemized",
)
# print(
#     context_chroma.similarity_search_with_score(
#         "jesus (s) speaks (v) to peter (o)", search_type="similarity", k=1
#     )
# )


# ## Get Syntax brackets

# os.system("pip install lxml")

# Get the plain treedown representation for a token's sentence

from lxml import etree
import requests

# Get the plain treedown representation for a token's sentence

# example endpoint: "https://labs.clear.bible/symphony-dev/api/GNT/Nestle1904/lowfat?usfm-ref=JHN%2014:1" - JHN 14:1


def process_element(element, usfm_ref, indent=0, brackets=False):
    if brackets:
        indent = 0
    treedown_str = ""
    open_bracket = "[" if brackets else ""
    close_bracket = "] " if brackets else ""

    if element.get("class") == "cl":
        treedown_str += "\n" + open_bracket + ("  " * indent)

    if element.get("role"):
        role = element.attrib["role"]
        if role == "adv":
            role = "+"
        if not brackets:
            treedown_str += "\n"
        treedown_str += open_bracket + ("  " * indent) + role + ": "

    # # bold the matching token using usfm ref # NOTE: not applicable, since I think you have to use a USFM ref without the word on the endpoint
    # if element.tag == "w" and element.get("ref") == usfm_ref:
    #     treedown_str += "**" + element.text + "**"
    #     treedown_str += element.attrib.get("after", "") + close_bracket

    if element.tag == "w" and element.text:
        treedown_str += (
            element.attrib.get("gloss", "")
            + element.attrib.get("after", "")
            + f"({element.text})"
        )
        treedown_str += close_bracket

    for child in element:
        treedown_str += process_element(child, usfm_ref, indent + 1, brackets)

    return treedown_str


def get_treedown_by_ref(usfm_ref, brackets=True):
    usfm_passage = usfm_ref.split("!")[0]
    endpoint = (
        "https://labs.clear.bible/symphony-dev/api/GNT/Nestle1904/lowfat?usfm-ref="
        # "http://localhost:8984/symphony-dev/api/GNT/Nestle1904/lowfat?usfm-ref="
        + usfm_passage
    )

    # uri encode endpoint
    endpoint = requests.utils.requote_uri(endpoint)

    # print(endpoint)

    text_response = requests.get(endpoint).text
    xml = etree.fromstring(text_response.encode("utf-8"))

    treedown = process_element(xml, usfm_passage, brackets=brackets)
    return treedown


def get_syntax_for_query(query):
    # Get passage using bible passage lookup and grabbing first result metadata['usfm']
    most_relevant_passage_usfm_ref = bible_chroma.search(
        query, search_type="similarity", k=1
    )[0].metadata["usfm"]
    return get_treedown_by_ref(most_relevant_passage_usfm_ref)


# Define callback handlers
# In[33]:


from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# # from langchain.callbacks.streamlit import StreamlitCallbackHandler
# from langchain.llms.fake import FakeListLLM

# from flask_socketio import emit
# from langchain.callbacks.base import BaseCallbackHandler
# from typing import Any


# class StreamingSocketIOCallbackHandler(BaseCallbackHandler):
#     """Callback handler for streaming. Only works with LLMs that support streaming."""

#     def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
#         """Run on new LLM token. Only available when streaming is enabled."""
#         emit("agent_stdout", {"stdout": token})


# from flask_socketio import emit
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from typing import Any, Dict, List, Union

# """Callback Handler that logs to streamlit."""
from typing import Any, Dict, List, Optional, Union

import streamlit as st

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class StreamlitSidebarCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to streamlit."""

    def __init__(self) -> None:
        """Initialize the callback handler."""
        # self.tokens_area = st.sidebar.markdown("## Database Resources Consulted")
        tokens_area = st.expander("See Agent Reasoning")
        self.tokens_area = tokens_area
        self.tokens_stream = ""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        # st.write("Prompts after formatting:")
        # for prompt in prompts:
        #     st.write(prompt)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.write(self.tokens_stream)
        # self.sidebar.write(self.tokens_stream)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        # class_name = serialized["name"]
        # st.sidebar.write(f"Entering new {class_name} chain...")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        # st.sidebar.write("Finished chain.")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Print out the log in specified color."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        # st.sidebar requires two spaces before a newline to render it
        st.markdown(action.log.replace("\n", "  \n"))

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        # verses_with_refs = output
        # try:
        #     # assuming output is your string
        #     matches = re.findall(
        #         r"metadata.: \{.source.: .(.*?)., .usfm.: ..*?.\}, page_content=.(.*?).",
        #         output,
        #     )

        #     # matches will be a list of tuples where the first element of the tuple is the source and the second element is the page_content
        #     verses_with_refs = [match[0] + ": " + match[1] + "\n" for match in matches]
        #     verses_with_refs = "  \n- ".join(verses_with_refs)

        #     # st.sidebar.markdown(verses_with_refs)
        # except Exception as e:
        #     st.write("error on tool end", e)
        with st.expander("See Sources"):
            st.markdown(
                f"**<span style='color:blue'>Checked these sources</span>:**\n{output}",
                unsafe_allow_html=True,
            )
        # st.write(f"output: /{output}/")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on text."""
        # st.sidebar requires two spaces before a newline to render it
        st.sidebar.write(text.replace("\n", "  \n"))

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        # st.sidebar requires two spaces before a newline to render it
        st.sidebar.write(finish.log.replace("\n", "  \n"))


# class StreamlitDropdownCallbackHandler(StreamlitSidebarCallbackHandler):
#     """Callback handler to put agent reasoning into a dropdown."""


class StreamingSocketIOCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        print('emitting emit("activity", {"loading": True})')
        emit("activity", {"loading": True})

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        print('emitting emit("activity", {"stdout": token})')
        emit("activity", {"stdout": token})

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        print('emitting emit("activity", {"loading": False})')
        emit("activity", {"loading": False})

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        print('emitting emit("activity", {"error": str(error)})')
        emit("activity", {"error": str(error)})

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        print('emitting emit("activity", {"tool_start": serialized})')
        emit("activity", {"tool_start": serialized})

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        print('emitting emit("activity", {"action": action.text})')
        emit("activity", {"action": action.text})

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        print('emitting emit("activity", {"result": finish, "loading": False})')
        emit("activity", {"result": finish, "loading": False})


import sys
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self) -> None:
        """Initialize the callback handler."""
        self.tokens_area = st.sidebar.empty()
        # self.tokens_area = tokens_area
        # self.tokens_area = st.empty()
        self.tokens_stream = "Agent Reasoning\n\n"

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()

        if token == ".":
            token = ".\n\n"

        self.tokens_stream += token
        # Replace 'Input:', 'Final Answer:', 'Action:', with '\n' + string in tokens_stream
        # self.tokens_stream = (
        #     self.tokens_stream.replace("Input:", "\nInput:")
        #     .replace("Final Answer:", "\nFinal Answer:")
        #     .replace("Action:", "\nAction:")
        # )

        # self.tokens_area.markdown(self.tokens_stream)

        if token == ".":
            token = ".\n\n"

        # Replace 'Input:', 'Final Answer:', 'Action:', with '\n' + string in tokens_stream
        formatted_tokens_stream = (
            self.tokens_stream.replace(
                "Action Input:", "\n<span style='color:red'>Action Input</span>"
            )
            .replace("Final Answer:", "\n<span style='color:green'>Final Answer</span>")
            .replace("I now know the final answer", "")
            .replace("Thought:", "\n<span style='color:orange'>Thought</span>")
            .replace("Action:", "\n<span style='color:blue'>Action</span>")
        )

        self.tokens_area.markdown(formatted_tokens_stream, unsafe_allow_html=True)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        # clear self.tokens_area

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        # tokens_area += f"OUTPUTS: {outputs}\n"

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        # st.write(f"Tool end output: {output}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""


# ## Define custom tools for an agent

from langchain.output_parsers import RegexParser
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)

prompt_template = """Use the following bible verses to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. \
Be careful to ensure that the answer is relevant to the question, and that the bible verses *actually* answer the question. Many Bible verses will be similar, but only because they are talking about the same topic. \

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
Helpful Answer: [answer here rephrasing and partially quoting the bible verse to demonstrate concisely how it answers the question]
Score: [score between 1 and 10]

Begin!

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=output_parser,
)
qa_rerank_chain = load_qa_with_sources_chain(
    OpenAI(temperature=0),
    chain_type="map_rerank",
    metadata_keys=["source"],
    return_intermediate_steps=True,
    prompt=PROMPT,
    verbose=True,
)

from langchain.tools import tool


@tool
def get_relevant_bible_verses(query: str) -> Dict[str, Any]:
    """Get relevant Bible verses for a query."""
    docs = bible_chroma.similarity_search(query, k=10)
    return qa_rerank_chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )


# Import things that are needed generically
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

document_qa_template = """\
Given the provided Bible passages and a question (which should be focused on exegetical or linguistic aspects), create a conclusive answer referencing the provided "SOURCES". 
If the answer is not found within the given sources, admit the inability to answer, but refrain from inventing answers.
ALWAYS include a "SOURCES" part in your response.

QUESTION: What is the implication of the term 'only begotten'?
=========
Content: For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life.
Source: John 3:16
=========
FINAL ANSWER: The term 'only begotten' is used to specify and clarify which Son is in view, pointing specifically to Jesus. 
SOURCES: John 3:16

QUESTION: Why isn't John 1:1 translated as 'a god was the Word'?
=========
Content: In the beginning was the Word, and the Word was with God, and the Word was God.
Syntax: [[p: In [the] (Ἐν)] beginning (ἀρχῇ)] [vc: was (ἦν)] [s: the (ὁ)] Word,(Λόγος)] and (καὶ)] 
[[s: the (ὁ)] Word (Λόγος)] [vc: was (ἦν)] [p: with (πρὸς)] - (τὸν)] God,(Θεόν)] and (καὶ)] 
[[p: God (Θεὸς)] [vc: was (ἦν)] [s: the (ὁ)] Word.(Λόγος)]
Source: John 1:1
=========
FINAL ANSWER: The term 'God' in John 1:1 functions as a verbal predicate (the 'p' role in the syntax data), not the subject. In English, we usually place the subject first. Also, Koine Greek doesn't have an indefinite article like 'a', so such an interpretation would have to be inferred from the context, and the rest of John's context may not support this interpretation.
SOURCES: John 1:1

QUESTION: Who was the first person Paul encountered in Rome?
=========
Content:
Source: 
=========
FINAL ANSWER: I couldn't find any documents related to that question, please modify your query or try something else. 
SOURCES: 

QUESTION: Is the concept of reincarnation compatible with Christianity?
=========
Content: It is appointed unto men once to die, but after this the judgment.
Source: Hebrews 9:27
Content: Truly, truly, I say to you, whoever hears my word and believes him who sent me has eternal life. He does not come into judgment, but has passed from death to life.
Source: John 5:24
=========
FINAL ANSWER: The passages I looked at seem to suggest that death is followed by judgment and not a cycle of rebirths, but this is really more of a theological question. For a more nuanced discussion, you would be better off consulting theological secondary sources directly.
SOURCES: Hebrews 9:27, John 5:24

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:
"""


# import term2md

# human_tool = load_tools(["human"])[0]

# For each chain, customize prompt with `chain.combine_documents_chain.llm_chain.prompt.template = document_qa_template`

from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    temperature=0,
    streaming=True,
    # callbacks=[StreamlitSidebarCallbackHandler(), StreamingStdOutCallbackHandler()],
    callbacks=[StreamingStdOutCallbackHandler()],
)

## To use the retrieval QA chain with the bible chroma:
bible_tool = RetrievalQAWithSourcesChain.from_chain_type(
    llm, chain_type="stuff", retriever=bible_chroma.as_retriever()
)
## To use the map-rerank chain with the bible chroma


context_retriever = context_chroma.as_retriever()
context_retriever.search_kwargs["distance_metric"] = "cos"
context_retriever.search_kwargs["fetch_k"] = 1
context_retriever.search_kwargs["maximal_marginal_relevance"] = False
context_retriever.search_kwargs["k"] = 1
# context_tool = RetrievalQAWithSourcesChain.from_chain_type(
#     llm, chain_type="map_rerank", retriever=context_retriever
# )
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
context_tool = qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="map_reduce", retriever=context_chroma.as_retriever()
)

theology_tool = RetrievalQAWithSourcesChain.from_chain_type(
    llm, chain_type="stuff", retriever=theology_chroma.as_retriever()
)
encyclopedic_tool = RetrievalQAWithSourcesChain.from_chain_type(
    llm, chain_type="stuff", retriever=encyclopedic_chroma.as_retriever()
)

# Update the prompts
bible_tool.combine_documents_chain.llm_chain.prompt.template = document_qa_template
# context_tool.combine_documents_chain.llm_chain.prompt.template = document_qa_template
theology_tool.combine_documents_chain.llm_chain.prompt.template = document_qa_template
encyclopedic_tool.combine_documents_chain.llm_chain.prompt.template = (
    document_qa_template
)

import pandas as pd

# from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent

macula_greek_verse_agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0),
    # mg, # verse_df (?)
    verse_df,
    # verbose=True,
)

macula_greek_words_agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0),
    # mg, # verse_df (?)
    mg,
    # verbose=True,
)

tools = [
    Tool(
        name="Bible Verse Reader Lookup",
        # Use the
        # func=lambda x: bible_chroma.search(x, search_type="similarity", k=2),
        # func=lambda x: bible_tool({"question": x}, return_only_outputs=True),
        # func=lambda x: get_relevant_bible_verses(x),
        func=lambda x: bible_chroma.search(x, search_type="similarity", k=10),
        description="useful for finding verses that are similar to the user's query, not suitable for complex queries. Be very careful to check whether the verses are actually relevant to the user's question and not just similar to the user's question in superficial ways",
        callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    Tool(
        name="Bible Words Lookup",
        func=macula_greek_words_agent.run,  # Note: using the NT-only agent here
        description="useful for finding information about individual biblical words from a Greek words dataframe, which includes glosses, lemmas, normalized forms, and more. This tool is not useful for grammar and syntax questions (about subjects, objects, verbs, etc.), but is useful for finding information about the words themselves",
    ),
    Tool(
        name="Bible Verse Dataframe Tool",
        func=macula_greek_verse_agent.run,  # Note: using the NT-only agent here
        description="useful for finding information about Bible verses in a bible verse dataframe in case counting, grouping, aggregating, or list building is required. This tool is not useful for grammar and syntax questions (about subjects, objects, verbs, etc.), but is useful for finding information about the verses (English or Greek or Greek lemmas) themselves",
        callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    Tool(
        name="Linguistic Data Lookup",
        func=lambda x: context_chroma.similarity_search(x, k=3),
        # func=lambda x: context_tool.run(x),
        # func=lambda query: get_similar_resource(context_chroma, query, k=2),
        # func=lambda x: context_tool({"question": x}, return_only_outputs=True),
        callbacks=[StreamlitSidebarCallbackHandler()],
        description="useful for finding answers about linguistics, discourse, situational context, participants, semantic roles (source/agent, process, goal, etc.), or who the speakers are in a passage. Input MUST ALWAYS include a scope keyword like 'discourse', 'roles', or 'situation'",
    ),
    # Tool(
    #     name="Context for Most Relevant Passage", # NOTE: this tool isn't working quite right. Needs some work
    #     func=get_context_for_most_relevant_passage.run,
    #     description="useful for when you need to find relevant linguistic context for a Bible passage. Input should be 'situation for' and the original user query",
    # callbacks=[StreamlitSidebarCallbackHandler()],
    # ),
    Tool(
        name="Syntax Data Lookup",
        func=lambda x: get_syntax_for_query(x),
        description="useful for finding syntax data about the user's query. Use this if the user is asking a question that relates to a sentence's structure, such as 'who is the subject of this sentence?' or 'what are the circumstances of this verb?'",
        callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    Tool(
        name="Theological Data Lookup",
        func=lambda x: theology_chroma.search(x, search_type="similarity", k=5),
        # func=lambda query: get_similar_resource(theology_chroma, query, k=2),
        # func=lambda x: theology_tool({"question": x}, return_only_outputs=True),
        callbacks=[StreamlitSidebarCallbackHandler()],
        description="if you can't find a linguistic answer, this is useful only for finding theological data about the user's query. Use this if the user is asking about theological concepts or value-oriented questions about 'why' the Bible says certain things. Always be sure to cite the source of the data",
    ),
    Tool(
        name="Encyclopedic Data Lookup",
        func=lambda x: encyclopedic_chroma.similarity_search(x, k=5),
        # func=lambda query: get_similar_resource(encyclopedic_chroma, query, k=2),
        # func=lambda x: encyclopedic_tool({"question": x}, return_only_outputs=True),
        callbacks=[StreamlitSidebarCallbackHandler()],
        description="useful for finding encyclopedic data about the user's query. Use this if the user is asking about historical, cultural, geographical, archaeological, or other types of information from secondary sources",
    ),
    Tool(
        name="Any Other Kind of Question Tool",
        func=lambda x: "Sorry, I don't know!",
        description="This tool is for vague, broad, ambiguous questions",
        callbacks=[StreamlitSidebarCallbackHandler()],
    ),
    # human_tool,
    # Tool(
    #     name="Get Human Input Tool",
    #     func=lambda x: input(x),
    #     description="This tool is for vague, broad, ambiguous questions that require human input for clarification",
    # ),
]

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools,
    llm,
    # OpenAI(
    #     temperature=0, streaming=True, callbacks=[StreamingSocketIOCallbackHandler()]
    # ),
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)


# # Flask UI - non streaming

# In[ ]:


# from flask import Flask, render_template, request
# from flask_socketio import SocketIO, emit

# app = Flask(__name__)
# socketio = SocketIO(app)


# def agent_run(inputs):
#     return agent.run(inputs)


# @app.route("/")
# def index():
#     return render_template("index.html")


# @socketio.on("run_agent")
# def handle_run_agent(message):
#     # Run your agent here
#     result = agent_run(message["data"])

#     # Emit the result to the client
#     emit("agent_result", {"result": result})


# if __name__ == "__main__":
#     socketio.run(app, port=5001)

# Flask UI - streaming

# from flask import Flask, render_template, request
# from flask_socketio import SocketIO, emit
# import sys
# from io import StringIO

# app = Flask(__name__)
# socketio = SocketIO(app)


# # Capture stdout
# class CapturedOutput:
#     def __enter__(self):
#         self._stdout = sys.stdout
#         sys.stdout = self._stringio = StringIO()
#         return self

#     def __exit__(self, type, value, traceback):
#         sys.stdout = self._stdout

#     def getvalue(self):
#         return self._stringio.getvalue()


# def agent_run(inputs):
#     with CapturedOutput() as output:
#         # Run the agent with the user's input
#         final_output = agent.run(inputs)
#     return final_output, output.getvalue()


# @app.route("/")
# def index():
#     return render_template("index.html")


# @socketio.on("run_agent")
# def handle_run_agent(message):
#     # Run your agent here
#     final_output, stdout = agent_run(message["data"])

#     # Stream the stdout to the client
#     for token in stdout.split():
#         emit("agent_stdout", {"stdout": token})

#     # Emit the final output to the client
#     emit("agent_result", {"result": final_output})


# if __name__ == "__main__":
#     socketio.run(app, port=5001)

## Flask streaming using socketio callback handler
# from flask import Flask, render_template, request
# from flask_socketio import SocketIO, emit
# import sys
# from io import StringIO
# import logging
# from time import sleep

# app = Flask(__name__)
# socketio = SocketIO(app)

# os.system(
#     "mkdir -p scripts && wget https://raw.githubusercontent.com/drudru/ansi_up/master/ansi_up.js -O scripts/ansi_up.js"
# )


# # Capture stdout
# class CapturedOutput:
#     def __enter__(self):
#         self._stdout = sys.stdout
#         sys.stdout = self._stringio = StringIO()
#         return self

#     def __exit__(self, type, value, traceback):
#         sys.stdout = self._stdout

#     def getvalue(self):
#         return self._stringio.getvalue()


# def agent_run(inputs):
#     with CapturedOutput() as output:
#         # Run the agent with the user's input
#         final_output = agent.run(inputs)
#     return final_output, output.getvalue()


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/log_stream")
# def stream_logs():
#     def generate():
#         with open("job.log") as f:
#             while True:
#                 yield f.read()
#                 sleep(1)

#     return app.response_class(generate(), mimetype="text/plain")


# @socketio.on("run_agent")
# def handle_run_agent(message):
#     # The callback handler will stream the output to the flask app.
#     agent_run(message["data"])


# if __name__ == "__main__":
#     logging.basicConfig(filename="job.log", level=logging.INFO)
#     socketio.run(app, port=5001, debug=True)

# Steamlit UI - Streaming version

# import streamlit as st
# import sys
# from io import StringIO
# from streamlit_pills import pills

# # Create a text input for user input
# user_input = st.text_input("Enter your message:")


# # Capture stdout
# class CapturedOutput:
#     def __enter__(self):
#         self._stdout = sys.stdout
#         sys.stdout = self._stringio = StringIO()
#         return self

#     def __exit__(self, type, value, traceback):
#         sys.stdout = self._stdout

#     def getvalue(self):
#         return self._stringio.getvalue()


# # Create a selection for streaming mode
# selected = pills("", ["NO Streaming", "Streaming"], ["🎈", "🌈"])

# if user_input:
#     if selected == "Streaming":
#         with CapturedOutput() as output:
#             # Run the agent with the user's input
#             final_output = agent.run(user_input)

#         # Display the final output in the main app
#         st.write(f"Chatbot: {final_output}")

#         # Stream the stdout in the sidebar
#         for token in output.getvalue().split():
#             st.sidebar.write(token)
#     else:
#         with CapturedOutput() as output:
#             # Run the agent with the user's input
#             final_output = agent.run(user_input)

#         # Display the final output in the main app
#         st.write(f"Chatbot: {final_output}")

#         # Display the stdout in the sidebar
#         st.sidebar.write(output.getvalue())

st.header("**Genesis** Exegetical Agent")
# st.image("/Users/ryderwishart/Downloads/DALL·E 2023-06-14 11.05.52.png", width=600)
from datetime import datetime

# Initialize the chat history in session_state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Create a container for the chat history
chat_container = st.expander("Question History")

# Create a container for the user input
input_container = st.container()


# Display the chat history
with chat_container:
    for sender, text in reversed(st.session_state["chat_history"]):
        is_user = sender == "user"
        sender_name = "User: " if is_user else "Genesis Exegetical Agent: "
        st.write(sender_name + text)

# Create a text input for user input in the input container
with input_container:
    user_input = st.text_input("Ask a question:")

    if user_input:
        chat_container.empty()  # clear the chat history widget
        ai_response = st.empty()
        st.empty()  # clear the input field

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Add the user's input to the chat history
        st.session_state["chat_history"].append(("user", user_input))

        st.write("User: " + user_input)

        # Run the agent with the user's input
        try:
            result = agent.run(input=user_input)
        except Exception as e:
            result = "Sorry, I don't know! I hit an error: " + str(e)

        # Add the agent's response to the chat history
        st.session_state["chat_history"].append(("agent", result))

        # Display the final output in the main app
        st.write("Genesis Exegetical Agent: " + result)


# st.header("**Genesis** Exegetical Agent")
# # st.image("/Users/ryderwishart/Downloads/DALL·E 2023-06-14 11.05.52.png", width=600)
# from streamlit_chat import message
# from datetime import datetime
# import random

# # Initialize the chat history in session_state if it doesn't exist
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []

# # Create a container for the chat history
# chat_container = st.expander("Question History")

# # Create a container for the user input
# input_container = st.container()


# # Display the chat history
# with chat_container:
#     for sender, text in reversed(st.session_state["chat_history"]):
#         is_user = sender == "user"
#         message(text, is_user=is_user, avatar_style="icons")

# # Create a text input for user input in the input container
# with input_container:
#     user_input = st.text_input("Ask a question:")

#     if user_input:
#         chat_container.empty()  # clear the chat history widget
#         ai_response = st.empty()
#         st.empty()  # clear the input field

#         timestamp = datetime.now().strftime("%H:%M:%S")

#         # Add the user's input to the chat history
#         st.session_state["chat_history"].append(("user", user_input))
#         message(
#             user_input,
#             is_user=True,
#             avatar_style="icons",
#             key=f"user-{timestamp}_{str(random.randint(0, 1000))}",
#         )

#         # Run the agent with the user's input
#         try:
#             result = agent.run(user_input)
#         except Exception as e:
#             result = "Sorry, I don't know! I hit an error: " + str(e)

#         # Add the agent's response to the chat history
#         st.session_state["chat_history"].append(("agent", result))

#         # Display the final output in the main app
#         message(
#             result,
#             # avatar_style="icons",
#             key=f"agent-{timestamp}_{str(random.randint(0, 1000))}",
#         )
