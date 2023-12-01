# You'll need to set the environment variables before running this
#
# export OPENAI_API_KEY=sk-y....
# export SERPAPI_API_KEY=dee1bd76e4...
#
# You can get the API keys from https://openai.com/ and https://serpapi.com/
#
# And then
#
# $ pip install langchain wikipedia openai google-search-results
#
# I'm not a Python programmer, so I'm sure there are better ways to do this
#
# RUN WITH EXTREME CAUTION, OR IN DOCKER,  IT HAS ACCESS TO YOUR TERMINAL
#

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.5)
tools = load_tools(['python_repl', 'requests', 'terminal', 'serpapi', 'wikipedia', 'human',  'pal-math', 'pal-colored-objects'], llm=llm)

agent = initialize_agent(tools, llm, agent="chat-zero-shot-react-description", verbose=True)

agent.run("Ask the human what they want to do")
