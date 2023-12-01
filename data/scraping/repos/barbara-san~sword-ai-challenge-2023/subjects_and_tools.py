# subjects
MATHS = "Mathematics"
HISTORY = "History"
PORTUGUESE = "Portuguese"
ENGLISH = "English (Secondary Language)"
PHY_CHEM = "Physics and Chemistry"
BIO_GEO = "Biology and Geology"
PHILOSOPHY = "Philosophy"

# import packages
from langchain.agents import Tool
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper, GoogleSerperAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

# load env. variables
from config import load_environment
load_environment()

# APIs
wolframalpha_api = WolframAlphaAPIWrapper()
wikipedia_api = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
duckduckgo_api = DuckDuckGoSearchRun()
google_serper_api = GoogleSerperAPIWrapper()

# tools
wolframalpha = Tool(
    name="Math Helper (WolframAlpha)",
    func=wolframalpha_api.run,
    description="useful for when you need to answer questions math related"
)
wikipedia = Tool(
    name="Wikipedia Helper",
    func=wikipedia_api.run,
    description="useful for when you need to search about any topic and get some historical knowledge from it through the information on Wikipedia"
)
duckduckgo = Tool(
    name="Web Search (DuckDuckGo)",
    func=duckduckgo_api.run,
    description="useful for when you need to search about any topic using the DuckDuckGo web engine"
)
google_serper = Tool(
    name="Intermediate Answer (Google Serper Search)",
    func=google_serper_api.run,
    description="useful for when you need to ask with search"
)

# tools mapping
TOOLS_OF = {
    MATHS : [wolframalpha],
    HISTORY : [wikipedia, duckduckgo, google_serper],
    PORTUGUESE : [wikipedia, duckduckgo, google_serper],
    ENGLISH : [wikipedia, duckduckgo, google_serper],
    PHY_CHEM : [wolframalpha, wikipedia, duckduckgo, google_serper],
    BIO_GEO : [wolframalpha, wikipedia, duckduckgo, google_serper],
    PHILOSOPHY : [wikipedia, duckduckgo, google_serper]
}