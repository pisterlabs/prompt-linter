import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from dotenv import load_dotenv
from operator_search import OperatorTool 
from langchain.agents import load_tools
from reservoir import ReservoirTool 
from langchain.callbacks import StreamlitCallbackHandler

load_dotenv()

# Load any required tools, like the 'requests_all' mentioned previously
requests_tools = load_tools(["requests_all"])

# Load the list of skills 
# Skills describe the types of questions the agent can answer, as well as how to answer them
with open('../prompts/skills.txt', 'r') as f:
    skills = f.read()

# Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

system_message = {
    f'''
    You are a helpful assistant tasked with answering questions. When encountering new words, do not attempt to change their spelling. Assume they are proper nouns.
    If the user input does not pertain to one of the following specific skills: {skills}, always respond with "Sorry, I can't answer that." Do not attempt to answer the question.
    '''
}

# Initialize agent with specific tools
tools = [OperatorTool(), ReservoirTool(),]

# Set the page config
st.set_page_config(
    page_title="NFT Assistant",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="collapsed"
)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Initialize the OpenAI model
llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-4',
    openai_api_key=openai_api_key
)

agent = initialize_agent(
    agent='structured-chat-zero-shot-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

# Main interface for user to ask a question
st.title("NFT Assistant")
with st.form(key="form"):
    user_input = st.text_input("Ask a question about a specific NFT project:")
    submit_clicked = st.form_submit_button("Submit Question")

combined_input = f"{user_input} {system_message}"
# Handle the response if a question is submitted
output_container = st.empty()
if submit_clicked:
    output_container = output_container.container()
    output_container.markdown(f"**Question:** {user_input}")
    
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    answer_container = output_container.markdown("**Answer:**", unsafe_allow_html=True)

    # Use the agent to generate an answer and display it
    st_callback = StreamlitCallbackHandler(answer_container)
    answer = agent.run(combined_input, callbacks=[st_callback])
    answer_container.markdown(f"<p>{answer}</p>", unsafe_allow_html=True)
