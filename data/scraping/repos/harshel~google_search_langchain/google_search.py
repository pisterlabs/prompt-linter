# Import necessary libraries
import streamlit as st
import os
from langchain.chains import ConversationChain
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.agents import AgentType, Tool, initialize_agent
from langchain import PromptTemplate
from langchain.llms import OpenAI
from datetime import datetime

# OpenAI API key
# Set API keys using Streamlit's secrets manager
os.environ["SERPAPI_API_KEY"] = "1d1187d187ce4aad8c840b13409d28baba0e385f5043f1b8fea305ee3fed5505"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "c6aaec012d15b23a05cc602d9e02e755f10d6672"

MODEL = "gpt-4"

# Set Streamlit page configuration
st.set_page_config(page_title='SearchBot', layout='wide')

# Define function to clear session state for a specific option
def clear_option_state(option):
    if option == "Search":
        st.session_state.search_session = {}
    elif option == "News":
        st.session_state.news_session = {}

# Initialize session states for each option
if "search_session" not in st.session_state:
    st.session_state.search_session = {}
if "news_session" not in st.session_state:
    st.session_state.news_session = {}

# Define function to get user input for a specific option
def get_text(option):
    if option == "Search":
        key = f"input_{option}"
        session_state = st.session_state.search_session
    elif option == "News":
        key = f"input_{option}"
        session_state = st.session_state.news_session
    else:
        key = "input_default"
        session_state = st.session_state

    return st.text_input(
        "You:",
        session_state.get("input", ""),
        key=key,  # Unique key for each option
        placeholder=f"Your {option.lower()} assistant here! Ask me anything ...",
        label_visibility="hidden",
    )

# Define function to clear session state for the current option
def clear_current_option_state():
    if st.session_state.option == "Search":
        st.session_state.search_session.clear()
    elif st.session_state.option == "News":
        st.session_state.news_session.clear()

def convert_date(date_str):
    try:
        return datetime.strptime(date_str, "%d %b %Y")
    except ValueError:
        return datetime.min

# Set Streamlit sidebar options
with st.sidebar:
    st.session_state.option = st.selectbox("Select the option?", ("Search", "News"))

# Define function to start a new chat for the current option
def new_chat():
    clear_current_option_state()

# Define the rest of your app logic for both "Search" and "News" options
######################### GENERAL SEARCH ##################################################
if st.session_state.option == "Search":
    st.title("Google Search")

    # Initialize session states
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []

    search = GoogleSerperAPIWrapper(type = "search")

    # Web Search Tool
    search_tool = Tool(
        name = "Web Search",
        func = search.run,
        description = "A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions.",
    )

    prompt = PromptTemplate(
    template="""Plan: {input}

    History: {chat_history}

    Let's think about answer step by step.
    If it's information retrieval task, solve it like a professor in particular field.""",
        input_variables = ["input", "chat_history"]
    )


    # Create an OpenAI instance
    llm = OpenAI(temperature = 0,
                    model_name = MODEL, 
                    verbose = False) 
        
    # Create a ConversationEntityMemory object if not already created
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)
       
    plan_chain = ConversationChain(
                llm = llm,
                memory = st.session_state.memory,
                input_key = "input",
                prompt = prompt,
                output_key = "output",
            )
    # Initialize Agent
    agent = initialize_agent(
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools = [search_tool],
            llm = llm,
            max_iterations = 3,
            prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory = st.session_state.memory
        )

    # # Get the user input
    # input_text = st.text_input("You: ", st.session_state["input"], key = "input1",
    #                         placeholder = "Your search assistant here! Ask me anything ...", 
    #                         label_visibility = 'hidden')

    # input_text = get_text(option = "Search")

    # Determine which session state to use
    current_option = st.session_state.option
    input_text = st.text_input(current_option)

    # Generate the output using the ConversationChain object and the user input, and add the input/output to the session
    if input_text:
        res = agent({"input": input_text, "chat_history": []})
        # # Agent execution
        response = res['output']
        st.session_state.past.append(input_text)  
        st.session_state.generated.append(response)

    with st.expander("Conversation", expanded = True):
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            st.info(st.session_state["past"][i], icon = "🧐")
            st.success(st.session_state["generated"][i], icon = "🤖")

##################### NEWS SEARCH #####################################
elif st.session_state.option == "News":
    st.title("News Search")

    # st.set_page_config(page_title = 'SearchBot', layout = 'wide')
    # Initialize session states
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []

    search = GoogleSerperAPIWrapper(type = "news")

    # # Get the user input
    # input_text = st.text_input("You: ", st.session_state["input"], key = "input2",
    #                         placeholder = "Your search assistant here! Ask me anything ...", 
    #                         label_visibility = 'hidden')

    # input_text = get_text(option = "News")

    # Determine which session state to use
    current_option = st.session_state.option
    input_text = st.text_input(current_option)

    if input_text:
        results = search.results(input_text)
        news_results = results['news']
        # Sort search results by date in descending order
        sorted_results = sorted(results['news'], key = lambda x: convert_date(x['date']), reverse = True)
        for result in sorted_results:
            st.write(f"# {result['title']}")
            st.write(f"**Date:** {result['date']}  |  **Source:** {result['source']}")
            
            if result.get('imageUrl'):
                st.image(result['imageUrl'], caption = "Image", use_column_width = False, width = 400)
            
            st.markdown(f"[**Link to Article**]({result['link']})")
            st.write(f"**Snippet:** {result['snippet']}")
            
            # Horizontal line separator
            st.markdown("---")

if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session

# Clear session state when switching between options
if st.session_state.option != current_option:
    clear_option_state(st.session_state.option)
    clear_current_option_state()

