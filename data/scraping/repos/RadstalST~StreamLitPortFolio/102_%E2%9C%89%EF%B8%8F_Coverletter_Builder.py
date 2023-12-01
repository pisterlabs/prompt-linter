import streamlit as st
import components.open_ai_key as open_ai_key
from dotenv import load_dotenv
import os

# langchain stack
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import ConversationChain,LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory,ConversationBufferWindowMemory
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

if not OPENAI_API_KEY:
    st.warning("Please enter your OpenAI API Key")
    st.stop()

temperature = st.slider("Temperature (model creativity)",0.0,1.0,0.05,0.1)
# AI building block

# 1. Prompt Template
execute_template = PromptTemplate(
    input_variables=["plan", "chat_history"],
    template="""
    Execute the plan step by step.
    and try to be short, concise, and clear.
    Plan: 
    {plan}

    HISTORY: 
    {chat_history}

    the final output should be a summary of the findings
    """
)
# improvement 1. split plan and summary
plan_creation_prompt_template = PromptTemplate(
    input_variables=["input"],
    template="""
    Prepare plan for task execution.

    Tools to use: wikipedia, web search

    REMEMBER: Keep in mind that you don't have information about current date, temperature, informations after September 2021. Because of that you need to use tools to find them.
    REMEMBER: You dont need an actual job posting link and you dont need any personal information.

    help me write a cover letter for a job application based on this job posting:

    {input}


    '''
# Execution plan: 

[execution_plan]

# Rest of needed information: 

[rest_of_needed_information]

    '''
    """
)
job_posting_and_resume_summary_template = PromptTemplate(
    input_variables=["input"],
    template="""
    Prepare summary of the job posting and my resume. 

    based on {input}
    
    in the following format:

    '''
    # Job posting summary:

    ## Job title: [job_title]

    ## Job requirements: [job_requirements]


    # Resume summary:

    ## Relevant experience: [relevant_experience]

    ## Relevant skills: [relevant_skills]

    ## Relevant education: [relevant_education]

    ## Relevant projects: [relevant_projects]

    ## Why I am a good fit: [why_i_am_a_good_fit]


    '''
    """)

coverletter_template = PromptTemplate(
    input_variables=["input","chat_history"],
    template="""
    summary:
    {input}
    Prepare cover letter for job application.
    Based on the chat history:
    {chat_history}

    and my resume 

    REMEMBER: dont make up any information, use only information from the chat history and my resume.
    Output should be a cover letter for a job application based on the job posting in markdown format.

    """
)

plan_summary_template = PromptTemplate(
    input_variables=["input","chat_history"],
    template="""

    {input}

    History:

    {chat_history}

    think about the plan and summary:
"""

)


search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()
llm = ChatOpenAI(temperature=temperature,openai_api_key=OPENAI_API_KEY)
tools = [
    Tool(
        name="Web Search",
        func=search.run,
        description="A web search tool is a software application or service that enables users to search for information on the internet. It is valuable for swiftly accessing a vast array of data and is widely used for research, learning, entertainment, and staying informed. With features like filters and personalized recommendations, users can easily find relevant results. However, web search tools may struggle with complex or specialized queries that require expert knowledge and can sometimes deliver biased or unreliable information. It is crucial for users to critically evaluate and verify the information obtained through web search tools, particularly for sensitive or critical topics.",
    ),
    Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="Wikipedia is an online encyclopedia that serves as a valuable web search tool. It is a collaborative platform where users can create and edit articles on various topics. Wikipedia provides a wealth of information on a wide range of subjects, making it a go-to resource for general knowledge and background information. It is particularly useful for getting an overview of a topic, understanding basic concepts, or exploring historical events. However, since anyone can contribute to Wikipedia, the accuracy and reliability of its articles can vary. It is recommended to cross-reference information found on Wikipedia with other reliable sources, especially for more specialized or controversial subjects.",
    )
]
# 2. Agent
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=6,return_messages=True)
plan_chain = LLMChain(
    llm=llm,
    prompt=plan_creation_prompt_template,
    output_key="plan",
)

job_posting_and_resume_summary_chain = LLMChain(
    llm=llm,
    prompt=job_posting_and_resume_summary_template,
    output_key="summary",
)
coverletter_chain = ConversationChain(
    llm=llm,
    memory=memory,
    input_key="input",
    prompt=coverletter_template,
    output_key="coverletter",
)
plan_summary_chain = ConversationChain(
    llm=llm,
    memory=memory,
    input_key="input",
    prompt=plan_summary_template,
    output_key="plan_summary",
)

agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    llm=llm,
    prompt_template=execute_template,
    max_iterations=5,
    tools=tools,
    memory=memory,
    )

@st.cache_data
def get_plan(input_str,temp=temperature):

    
    plan = plan_chain.run(
        input = input_str,
    )
    return plan

@st.cache_data
def get_job_posting_and_resume_summary(input_str,temp=temperature):
    ret = job_posting_and_resume_summary_chain.run(
        input = input_str,
    )
    return ret
@st.cache_data
def get_answer(plan,temp=temperature):
    ret = agent(plan)
    return ret

@st.cache_data
def get_coverletter(resume_and_job_posting,temp=temperature):
    ret = coverletter_chain.run(
        input = resume_and_job_posting,
    )
    return ret

@st.cache_data
def get_plan_summary(plan_summary,temp=temperature):
    ret = plan_summary_chain.run(
        input = plan_summary
    )

    return ret
st.title("🦾 AI Agent Coverletter Builder")


with st.form(key="form"):
    input_job_description = st.text_area("Enter the job description here")
    input_resume = st.text_area("Enter your resume here")


    submit_button = st.form_submit_button(label="Submit")


if submit_button:
    input_str = f"""
        {input_job_description}

        and my resume:
        {input_resume}
    """
    # run AI agent
    plan = get_plan(input_str,temp=temperature)
    with st.expander("Plan"):
        st.write(plan)


    summary = get_job_posting_and_resume_summary(input_str,temp=temperature)


    
    with st.expander("Summary"):
        st.write(summary)
    plan_summary = get_plan_summary(f"""
    {plan}
    {summary}
    """,temp=temperature)
    with st.expander("Plan Summary"):
        st.write(plan_summary)

    plan_excution = get_answer(plan,temp=temperature)
   
    with st.expander("Plan Execution",expanded=False):
        st.write(plan_excution)
        

    coverletter = get_coverletter(summary,temp=temperature)
    with st.expander("Coverletter",expanded=True):
        st.markdown(coverletter)
