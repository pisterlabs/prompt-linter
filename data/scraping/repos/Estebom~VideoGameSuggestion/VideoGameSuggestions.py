#Esteban Rodriguez
#11/15/2023

#imports
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools, AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
import pandas as pd
import os

#Setting up title
st.title("Video Game Suggestions")

#Setting up llm parameters and agent
llm = OpenAI(temperature=0.9, streaming=True)
tools = load_tools(['ddg-search'])
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

#options for each option 
df = pd.DataFrame({
    'Genre' : ['Select a Genre', 'RPG', 'Shooter', 'Adventure', 'Story'],
    'Players' : ['Select Player Choice', 'Single-Player', 'Multiplayer', 'Co-op', 'MMO'],
    'Price' : ['Select a Price Option', 'Live-Service', 'Free-to-Play', 'Pay Once', 'Micro-Transactions']
})

#selectboxes for game categories
option_1 = st.selectbox('Pick your Genre', df['Genre'], index=0)
option_2 = st.selectbox('Pick your player choice', df['Players'], index=0)
option_3 = st.selectbox('Pick your price', df['Price'], index=0)

#text box for extra details
prompt = st.text_input("Please add any extra details for the games you like:")

if 'first_submit' not in st.session_state:
    st.session_state['first_submit'] = False

#hit submit once ready
if st.button('Submit', key='Submit1'):
    st.session_state['first_submit'] = True
    if option_1 != 'Select a Genre' and option_2 != 'Select Player Choice' and option_3 != 'Select a Price Option':
        with st.container():
            st_callback = StreamlitCallbackHandler(st.container())
            #prompt passed into llm and duckduckgo as well as formating for output
            response = agent.run(llm.invoke(f"Please recommend 5 games that align with these features: "
                                f"Genre: {option_1}, "
                                f"Players that can play: {option_2}, "
                                f"Price of game: {option_3}. "
                                f"If they meet any of these additional details requests: '{prompt}'. "
                                f"Please give a summary of what each game is and what the gameplay is like."),
                     callbacks=[st_callback])
            st.write(response)
            st.session_state['first_response'] = response
    #Case if categories are not picked
    else:
        st.error("Please make a selection in all categories.")

if 'first_response' in st.session_state:
    actual_response = st.session_state['first_response']
else:
    actual_response = "No response yet"

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = ""

#prompt 
prefix = f"""Answer the questions pertaining to the following games listed : {actual_response}"""
suffix= """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""


prompt2 = ZeroShotAgent.create_prompt(

    tools, 
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

#llm chain and agent creation
llm_chain = LLMChain(llm = OpenAI(temperature = 0.9), prompt=prompt2)
agent2 = ZeroShotAgent(llm_chain=llm_chain, tools = tools, verbose = True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent = agent2 , tools=tools, verbose = True, memory = memory
)


if st.session_state['first_submit']:
    input = st.text_input("Feel free to ask any questions about the games listed:")
    if st.button('Submit', key = 'Submit2'):
        st.session_state['chat_history'] += f"Q: {input}\n"
        if st.session_state['chat_history'].strip() == f"Q: {input}\n":
            # first interaction after selections
            prefix = f"""Answer the questions pertaining to the following games listed: {actual_response}"""
        else:
            # ongoing conversation
            prefix = "Answer the questions based on the ongoing conversation:"

        # recreats prompt pased on new prefix
        prompt2 = ZeroShotAgent.create_prompt(
            tools, 
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )

        #handles output flow
        st.chat_message("user").write(input)
        with st.chat_message("assistant"):
            st_callback2 = StreamlitCallbackHandler(st.container())
            response2 = agent_chain.run({"input": input, "chat_history": st.session_state['chat_history']}, callbacks=[st_callback2])     
            st.write(response2) 

            st.session_state['chat_history'] += f"A: {response2}\n"

        # clears input field
        st.session_state['question_input'] = ""
           

