import streamlit as st
import components.open_ai_key as open_ai_key
from dotenv import load_dotenv
import os

# langchain stack
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import LLMMathChain


load_dotenv()
try:
    #if set in env
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    #if set but empty
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = open_ai_key.render()
except:
    #if not set
    OPENAI_API_KEY = open_ai_key.render()


st.title("🦾 AI Agent demo")
st.write("""
This is a demo of the AI Agent that I built using OpenAI's API and LangChan. 
## capabilities
- The AI Agent can answer questions based on the information on the internet by using DuckDuckGo's API and Wikipedia's API.
- The AI Agent can plan and execute a plan to solve a problem.
- The AI Agent do math.
""")
st.write("""
>This demo requires an OpenAI API Key.
""")
         
with st.expander("Inner Workings"):
    st.write("""

## Inner Workings
The AI agent works by dividing the task into two parts: planning and execution.
### 1. Planning
the AI agent uses the planning model to plan the steps to solve the problem. In this case 
```python
planner = load_chat_planner(planner_model,)
```

### 2. Execution
the AI agent uses the execution model to execute the plan. In this case
```python
executor = load_agent_executor(llm, tools, verbose=True)
```
the load_agent_executor function loads the agent executor
which are able to use the provided tools:
- Web Search > DuckDuckGo
- Wikipedia > Wikipedia
- Calculator > LLMMathChain

### 3. Putting it all together
```python
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
```

## Conclusion
The AI agent is able to answer questions based on the information on the internet by using DuckDuckGo's API and Wikipedia's API.

""")
if OPENAI_API_KEY == "":
    st.warning("Please enter your OpenAI API Key in the sidebar.")
    st.stop()
else:
    st.info("OpenAI API Key is set.")

#define model
llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo",openai_api_key=OPENAI_API_KEY)
planner_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",openai_api_key=OPENAI_API_KEY)
llm_math = LLMMathChain.from_llm(llm, verbose=True)
search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()
# define tools
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A web search tool is a software application or service that enables users to search for information on the internet. It is valuable for swiftly accessing a vast array of data and is widely used for research, learning, entertainment, and staying informed. With features like filters and personalized recommendations, users can easily find relevant results. However, web search tools may struggle with complex or specialized queries that require expert knowledge and can sometimes deliver biased or unreliable information. It is crucial for users to critically evaluate and verify the information obtained through web search tools, particularly for sensitive or critical topics.",
)
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="Wikipedia is an online encyclopedia that serves as a valuable web search tool. It is a collaborative platform where users can create and edit articles on various topics. Wikipedia provides a wealth of information on a wide range of subjects, making it a go-to resource for general knowledge and background information. It is particularly useful for getting an overview of a topic, understanding basic concepts, or exploring historical events. However, since anyone can contribute to Wikipedia, the accuracy and reliability of its articles can vary. It is recommended to cross-reference information found on Wikipedia with other reliable sources, especially for more specialized or controversial subjects.",
)
math_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="useful for when you need to answer questions about math"
)
tools = [search_tool, wikipedia_tool,math_tool]
# build AI agent
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
planner = load_chat_planner(planner_model,)
executor = load_agent_executor(llm, tools, verbose=True)
agent = PlanAndExecute(
    planner=planner, 
    executor=executor, 
    verbose=True,
    max_iterations=5)

@st.cache_data
def get_answer(question):
    return agent.run(question)



with st.form(key="form"):
    question = st.text_input("Question:",value="What is Apple Vision Pro Price divided by 0.4 ?")
    submit = st.form_submit_button("Ask")


if submit:
    with st.spinner("Answering..."):
        answer = get_answer(question)
    st.success(answer)


