import asyncio
from dataclasses import dataclass
from pydantic import BaseModel, Extra, Field
from typing import Any, Awaitable, Callable, ClassVar, Coroutine, Dict, List, Optional, Type
import logging



from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler

from langchain import LLMChain, PromptTemplate
from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.tools import BaseTool as LangChainBaseTool
from langchain.tools.base import ToolException
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.base import BaseChatModel
from langchain.base_language import BaseLanguageModel
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from arxiv import Result

from ai.cite import get_cites, scholar_lookup
from ai.store import PaperStore
from ai.arxiv import ArxivFetch, LoadedPapersStore, PaperMetadata
from ai.prompts import ABSTRACT_QS_PROMPT, ABSTRACT_SUMMARY_PROMPT, MAP_PROMPT, MULTI_QUERY_PROMPT, REDUCE_COMPREHENSIVE_PROMPT, REDUCE_KEYPOINTS_PROMPT, REDUCE_LAYMANS_PROMPT, SEARCH_TOOL

@dataclass
class PaperBackend:
    """
    Allows tools to refer to common objects. 
    Specifically the chat_id to track mentioned papers in a chat. Is inserted into pre-prompt for better tool use
    """
    chat_id: str            # can track mentioned papers for a chat, for better tool use and easier prompting
    vectorstore: Chroma     # for getting, inserting, filtering, document embeddings
    paper_store: PaperStore # paper metadata: title, abstract, generated summaries
    llm: BaseLanguageModel  # for various Chains


class BaseTool(LangChainBaseTool):
    """Lets tools define a user friendly action text to be displayed in progress updates"""
    action_label: str

class BasePaperTool(BaseTool):
    """Base class for tools which may want to load a paper before running their function."""

    _backend: Optional[PaperBackend]
    _text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    ) 

    class Config:
        extra = Extra.allow

    # aliases to backend objects for subclasses
    def llm(self):
        return self._backend.llm
    
    def paper_store(self):
        return self._backend.paper_store
    
    def vectorstore(self):
        return self._backend.vectorstore
    
    def set_backend(self, backend: PaperBackend):
        self._backend = backend


    def load_paper(self, paper_id: str) -> bool:
        """Load a paper. Will download if it doesn't exist in vectorstore.
        return: Whether it was already in the vectorstore."""
        if self._backend is None:
            raise Exception(f"No paper backend to load paper `{paper_id}`")
        
        # check for existing Docs of this paper
        result = self._backend.vectorstore.get(where={"source":paper_id})
        if len(result["documents"]) != 0: # any key can be checked
            found = True # already in db
        else:
            doc, abstract = arxiv_fetch.get_doc_sync(paper_id)
            self._backend.paper_store.save_title_abstract(paper_id, doc.metadata["title"], abstract)

            # split and embed docs in vectorstore
            split_docs = self._text_splitter.split_documents([doc])
            self._backend.vectorstore.add_documents(split_docs)
            found = False

        self._backend.paper_store.add_mentioned_paper(paper_id, self._backend.chat_id)
        return found
    
    async def aload_paper(self, paper_id: str) -> bool:
        """Load a paper. Will download if it doesn't exist in vectorstore.
        return: Whether it was already in the vectorstore."""
        if self._backend is None:
            raise Exception(f"No paper backend to load paper `{paper_id}`")
        
        # check for existing Docs of this paper
        result = self._backend.vectorstore.get(where={"source":paper_id})
        if len(result["documents"]) != 0: # any key can be checked
            found = True # already in db
        else:
            doc, abstract = await arxiv_fetch.get_doc_async(paper_id)
            self._backend.paper_store.save_title_abstract(paper_id, doc.metadata["title"], abstract)
            
            # split and embed docs in vectorstore
            split_docs = self._text_splitter.split_documents([doc])
            self._backend.vectorstore.add_documents(split_docs) # TODO: find store with async implementation
            found = False

        self._backend.paper_store.add_mentioned_paper(paper_id, self._backend.chat_id)
        return found

arxiv_fetch = ArxivFetch() # arxiv API wrapper

TOOL_ACTIONS = {}
def register_tool_action(cls: BaseTool):
    """A class decorator to track all tools, create a mapping which stores tool action labels"""
    TOOL_ACTIONS[cls.name] = cls.action_label

class ArxivSearchSchema(BaseModel):
    query: str = Field(description="arXiv search query. Refuse user queries which are to vague.")


class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = SEARCH_TOOL
    args_schema: Type[ArxivSearchSchema] = ArxivSearchSchema

    action_label = "Searching arxiv"

    def format_result(self, result: Result):
        abstract = result.summary[:200].replace('\n', '')
        return f"- [`{ArxivFetch._short_id(result.entry_id)}`] - `{result.title}`\n    - {abstract}..."

    def _run(self, query: str) -> List[str]:
        return "\n".join([self.format_result(r) for r in arxiv_fetch.search_async(query)])
    
    async def _arun(self, query: str):
        return "\n".join([self.format_result(r) for r in await arxiv_fetch.search_async(query)])

class PaperQASchema(BaseModel):
    question: str = Field(description="A question to ask about a paper. Cannot be empty. Do not include the paper ID")
    paper_id: str = Field(description="ID of paper to query")


class PaperQATool(BasePaperTool):
    name = "paper_question_answering"
    description = "Ask a question about the contents of a paper. Primary source of factual information for a paper. Don't include paper ID/URL in the question."
    args_schema: Type[PaperQASchema] = PaperQASchema

    action_label = "Querying a paper"
    # Uses LLM for QA retrieval chain prompting
    # Vectorstore for embeddings of currently loaded PDFs
    
    def _run(self, question, paper_id) -> str:
        self.load_paper(paper_id)
        qa = self._make_qa_chain(paper_id)
        return qa.run(question)


    async def _arun(self, question, paper_id) -> str:
        await self.aload_paper(paper_id) 
        qa = self._make_qa_chain(paper_id)
        return qa.run(question)


    def _make_qa_chain(self, paper_id: str):
        """Make a RetrievalQA chain which filters by this paper_id"""
        filter = {
            "source": paper_id
        }
        
        retriever = self.vectorstore().as_retriever(search_kwargs={"filter": filter})
        # TODO: generate multiple queries from different perspectives to pull a richer set of Documents
            
        qa = RetrievalQA.from_chain_type(
            llm=self.llm(),
            chain_type="stuff",
            retriever=retriever
        )
        return qa
    

class AbstractSummarySchema(BaseModel):
    paper_id: str = Field(description="arXiv paper ID")
    
class AbstractSummaryTool(BasePaperTool):
    name = "abstract_summary"
    description = "Returns a bullet point summary of the abstract. Do not modify this tool's output in your response. Use specifically when a short summary is needed."
    args_schema: Type[AbstractSummarySchema] = AbstractSummarySchema
    
    action_label = "Summarizing abstract"

    def _run(self, paper_id):
        # TODO: use threading map
        raise NotImplementedError("Use async version `_arun`")
    

    async def _arun(self, paper_id):
        await self.aload_paper(paper_id)

        abstract = self.paper_store().get_abstract(paper_id)
        title = self.paper_store().get_title(paper_id)

        # TODO: should be an instance variable since unchanged
        # summarize the abstract highlighting key points
        abs_summary_chain = LLMChain(prompt=ABSTRACT_SUMMARY_PROMPT, llm=self.llm())
        abs_summary = await abs_summary_chain.arun(title=title, abstract=abstract)
        
        return abs_summary
        

class SummarizePaperSchema(BaseModel):
    paper_id: str = Field(description="arXiv paper id")
    # Need to mention default values in natural language
    # since OpenAI function calling JSON does not specify exactly what the default value is.
    # This means the LLM cannot reason about whether the default value or another value is more appopriate
    type: str = Field(description="Type of summary. One of: {keypoints, laymans, comprehensive}, default to keypoints")


class SummarizePaperTool(BasePaperTool):
    name = "summarize_paper_full"
    description = "Summarizes a paper in full, with significant detail."

    args_schema: Type[SummarizePaperSchema] = SummarizePaperSchema

    action_label = "Summarizing entire paper"
    
    _summary_prompt = {
        "keypoints": REDUCE_KEYPOINTS_PROMPT, 
        "laymans": REDUCE_LAYMANS_PROMPT,
        "comprehensive": REDUCE_COMPREHENSIVE_PROMPT
    }

    def _run(self, paper_id, type):
        try:
            combine_prompt = self._summary_prompt[type]
        except KeyError:
            raise ToolException(f"Unknown summary type: \"{type}\"")
        self.load_paper(paper_id)

        existing_summary = self.paper_store().get_summary(paper_id, type)
        if existing_summary:
            return existing_summary

        map_reduce_chain = load_summarize_chain(
            llm=self.llm(), 
            chain_type="map_reduce", 
            map_prompt=MAP_PROMPT,
            combine_prompt=combine_prompt
        )
            
        result = self.vectorstore().get(where={"source": paper_id})
        chunks = result["documents"]
        if len(chunks) == 0:
            raise ToolException("Document not loaded or does not exist")
        
        docs = [Document(page_content=chunk) for chunk in chunks]
        summary = map_reduce_chain.run(docs)

        self.paper_store().save_summary(paper_id, type, summary)
        return summary

    async def _arun(self, paper_id, type):
        try:
            combine_prompt = self._summary_prompt[type]
        except KeyError:
            raise ToolException(f"Unknown summary type: \"{type}\"")
        await self.aload_paper(paper_id)

        existing_summary = self.paper_store().get_summary(paper_id, type)
        if existing_summary:
            return existing_summary

        map_reduce_chain = load_summarize_chain(
            llm=self.llm(),
            chain_type="map_reduce", 
            map_prompt=MAP_PROMPT,
            combine_prompt=combine_prompt
        )
            
        result = self.vectorstore().get(where={"source": paper_id})
        chunks = result["documents"]
        if len(chunks) == 0:
            raise ToolException("Document not loaded or does not exist")
        
        docs = [Document(page_content=chunk) for chunk in chunks]
        summary = await map_reduce_chain.arun(docs)

        self.paper_store().save_summary(paper_id, type, summary)
        return summary


class PaperCitationsTool(BaseTool):
    name = "get_citations"
    description = "Get a list of citations of an arXiv paper."
    
    action_label = "Searching for citations"

    async def _arun(self, paper_id: str):
        cite_id = await scholar_lookup(paper_id) # TODO: scrape or use another SerpAPI request
        citations = await get_cites(cite_id)
        return citations

class AbstractQuestionsSchema(BaseModel):
    paper_id: str = Field(description="ID of paper.")

class AbstractQuestionsTool(BasePaperTool):
    name = "get_abstract_questions"
    description = "Generates a set of questions to jump start discussion of a paper. Uses the paper's abstract."
    args_schema: Type[AbstractQuestionsSchema] = AbstractQuestionsSchema

    action_label = "Generating abstract questions"

    def _run(self, paper_id):
        self.load_paper(paper_id)
        abstract = self.paper_store().get_abstract(paper_id)
        title = self.paper_store().get_title(paper_id)

        # TODO: should be an instance variable since unchanged
        llm_chain = LLMChain(prompt=ABSTRACT_QS_PROMPT, llm=self.llm())
        
        return llm_chain.run(title=title, abstract=abstract)
    
    async def _arun(self, paper_id):
        await self.aload_paper(paper_id)
        abstract = self.paper_store().get_abstract(paper_id)
        title = self.paper_store().get_title(paper_id)

        # TODO: should be an instance variable since unchanged
        llm_chain = LLMChain(prompt=ABSTRACT_QS_PROMPT, llm=self.llm())
        
        return await llm_chain.arun(title=title, abstract=abstract)


class HumanFeedbackSchema(BaseModel):
    question: str = Field(description="A question about function arguments, whether the function call is appropriate.")
class HumanFeedbackTool(BaseTool):
    name = "human_feedback"
    description = "Ask the user for help when you are unsure about function arguments, have made assumptions about function inputs or want to make sure the function call is what's wanted. Use this before any other tool when unsure."
    args_schema: Type[HumanFeedbackSchema] = HumanFeedbackSchema

    action_label = "Asking for user input"

    def _run(self, question: str):
        return input(question + "\n>>> ")
    async def _arun(self, question: str):
        return input(question + "\n>>> ")



class FiveKeywordsTool(BaseTool):
    pass


class ConversationQuestionsTool(BaseTool):
    """Let LLM pass a conversation history to this tool and generate future "conversation fueling" questions."""
    pass





def get_tools(parse_tool_error: Optional[Callable[[ToolException], str]] = None, backend: Optional[PaperBackend] = None) -> list[BaseTool]:
    """Tools. Optionally initialize with a backend/error handler if used for execution in AgentExecutor.
    Otherwise, serves as a template when initializing Agent classes."""
    arxiv_search = ArxivSearchTool()
    # paper_citations = PaperCitationsTool()
    human_feedback = HumanFeedbackTool()
    
    # tools with paper backend
    abs_summary = AbstractSummaryTool()
    abs_questions = AbstractQuestionsTool()
    paper_qa = PaperQATool(handle_tool_error=parse_tool_error)
    paper_summary = SummarizePaperTool(handle_tool_error=parse_tool_error)

    tools = [arxiv_search, abs_summary, abs_questions, paper_qa, paper_summary, human_feedback]
    
    # register backend for tools that need it
    if backend is not None:
        for tool in tools:
            if isinstance(tool, BasePaperTool):
                tool.set_backend(backend)

    return tools


class DiscordToolCallback(BaseCallbackHandler):
    # args: message, is_done 
    Callback = Callable[[str, bool], Awaitable[None]]
    
    def __init__(self, handle_tool_msg: Callback, tools: list[BaseTool]):
        super().__init__()
        self.handle_tool_msg = handle_tool_msg
        self.tools = tools

    async def on_tool_start(
        self, 
        serialized: Dict[str, Any], 
        input_str: str,
        **kwargs
    ):
        print("=====")
        print(serialized)
        print(input_str)
        print("=====")
        print(input_str)
        print(kwargs)
        print("=====")

        tool_name = serialized["name"]
        to_action = {tool.name: tool.action_label for tool in self.tools}
        await self.handle_tool_msg(to_action[tool_name], False)

    async def on_tool_end(
        self, 
        output: str,
        **kwargs
    ):
        await self.handle_tool_msg("", True)