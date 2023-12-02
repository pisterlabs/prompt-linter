##### search
from langchain.utilities import GoogleSerperAPIWrapper
search_google_serp = GoogleSerperAPIWrapper()

from langchain.utilities import SerpAPIWrapper
search_serp = SerpAPIWrapper()

from langchain.utilities import GoogleSearchAPIWrapper
search_goolge = GoogleSearchAPIWrapper()

from langchain.tools import DuckDuckGoSearchRun
search_duck = DuckDuckGoSearchRun()


##### wiki
from langchain.docstore import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
docstore_wiki = DocstoreExplorer(Wikipedia())

from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
wiki_query = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


##### wolfram
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
wolfram_alpha = WolframAlphaAPIWrapper()


##### llm related
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
from langchain.chains import LLMMathChain
llm_math = LLMMathChain.from_llm(
    llm=OpenAI(temperature=0),
    verbose=True
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from module.query_vdb import get_faiss_multi_query_retriever, get_faiss_vdb_retriever
from pathlib import Path
_pwd = Path(__file__).absolute()
_vdb_path = _pwd.parent.parent.parent

### faiss_azure
_azure = str(_vdb_path / "vdb" / "azure_vm")
# _retriever_azure = get_faiss_vdb_retriever(_azure)
_retriever_azure = get_faiss_multi_query_retriever(_azure)
retriev_azure = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0),
    retriever=_retriever_azure
)

### faiss_langchain
# _langchain = str(_vdb_path / "vdb" / "langchain_python_documents")
# # _retriever_langchain = get_faiss_vdb_retriever(_langchain)
# _retriever_langchain = get_faiss_multi_query_retriever(_langchain)
# retriev_langchain = RetrievalQA.from_chain_type(
#     llm=ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0),
#     retriever=_retriever_langchain
# )


##### pubmed
from langchain.tools import PubmedQueryRun
pubmed_query = PubmedQueryRun()


##### youtube
from langchain.tools import YouTubeSearchTool
youtube_search = YouTubeSearchTool()


##### human
def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)

from langchain.tools import HumanInputRun
human_input = HumanInputRun(input_func=get_input)


##### weather
from langchain.utilities import OpenWeatherMapAPIWrapper
weather_map = OpenWeatherMapAPIWrapper()


##### file sys
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.agents.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory

### We'll make a temporary directory to avoid clutter
working_directory = TemporaryDirectory()
file_toolkit = FileManagementToolkit(root_dir=str(working_directory.name)).get_tools()
cp_file = file_toolkit[0]
del_file = file_toolkit[1]
search_file = file_toolkit[2]
mv_file = file_toolkit[3]
read_file = file_toolkit[4]
write_file = file_toolkit[5]
list_dir = file_toolkit[6]

