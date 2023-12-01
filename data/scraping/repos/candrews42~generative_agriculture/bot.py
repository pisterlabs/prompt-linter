# Import required libraries
import os
import utils
import streamlit as st
from langchain.agents import create_sql_agent, AgentExecutor, load_tools, AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from streaming import StreamHandler
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import sqlalchemy
from bot_instructions import chatbot_instructions, sqlbot_instructions
from langchain.chains import LLMChain, SequentialChain

# Streamlit page setup
st.set_page_config(page_title="GenAg Chatbot", page_icon="🌱") #, layout="wide")
st.header("Generative Agriculture Chatbot")
st.write("Natural language farm management tool")

# Define the main class for the Generative Agriculture Chatbot
class GenerativeAgriculture:
    # Initialize chatbot settings and API keys
    def __init__(self):
        utils.configure_openai_api_key()
        #self.openai_model = "gpt-3.5-turbo-instruct"
        self.openai_model = "gpt-3.5-turbo"
        #self.openai_model = "gpt-4-0613"
        #self.openai_model = "gpt-4-32k" # 4x context length of gpt-4

    # Setup database and agent chain
    @st.cache_resource
    def setup_chain(_self):
        # Database Connection
        username, password, host, port, database = [st.secrets[key] for key in ["username", "password", "host", "port", "database"]]
        db_url = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
        db = SQLDatabase.from_uri(db_url)

        # Initialize Database Connection
        try:
            engine = sqlalchemy.create_engine(db_url)
            conn = engine.connect()
        except Exception as e:
            st.write(f"An error occurred: {e}")
            exit()
        
        # Initialize memory setup (commented out for future use)
        chatbot_memory = ConversationBufferWindowMemory(k=5)
        # sqlagent_memory = ConversationBufferMemory()

        # Setup Chatbot
        chatbot_prompt_template = PromptTemplate(
            input_variables = ['user_input'],
            template=chatbot_instructions
        )
        llm=OpenAI(model_name=_self.openai_model, temperature=0.0, streaming=True)
        chatbot_agent = LLMChain(
            llm=llm, 
            memory=chatbot_memory, 
            prompt=chatbot_prompt_template, 
            verbose=True)
        
        # Setup SQL toolkit and agent
        toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))
        sql_agent = create_sql_agent(
            llm=OpenAI(temperature=0.1, streaming=True),
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
        return chatbot_agent, sql_agent
    
    # Main function to handle user input and chatbot response
    @utils.enable_chat_history
    def main(self):
        chatbot_agent, sql_agent = self.setup_chain()
        user_query = st.chat_input(placeholder="Enter your observation or question about the farm")
        sql_response = None
        if user_query:
            utils.display_msg(user_query, 'user')
            # TODO: Add user_query to raw_observations table
            # try:
            #     conn.execute(sql_query)
            #     st.write("Query executed successfully.")
            # except Exception as e:
            #     st.write(f"An error occurred: {e}")
                
            with st.chat_message("assistant"):
                # TODO run the below query to add user_query to raw_observations table
                # raw_observation = f"INSERT INTO raw_observations (observation) VALUES ('{user_query}');"
                st_cb = StreamHandler(st.empty())
                #formatted_user_query = chatbot_instructions.format(user_input=user_query)
                chatbot_response = chatbot_agent.run(user_query, callbacks=[st_cb])
                st.session_state.sql_query_to_run = chatbot_response
                st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
                # TODO if streamlit button pressed "Run Query", run the below query using the sql agent
            sql_button = st.button('Execute SQL Query')
            if sql_button:
                with st.chat_message("assistant"):
                    st_cb = StreamHandler(st.empty())
                    #formatted_user_query = chatbot_instructions.format(user_input=user_query)
                    sql_response = sql_agent.run(chatbot_response, callbacks=[st_cb])
                    # sql_response = sql_agent.run(st.session_state.sql_query_to_run, callbacks=[st_cb])
                    st.session_state.messages.append({"role": "assistant", "content": sql_response})
                    st.write(sql_response)
                    # TODO if streamlit button pressed "Run Query", run the below query using the sql agent
                
# Entry point of the application
if __name__ == "__main__":
    obj = GenerativeAgriculture()
    obj.main()
