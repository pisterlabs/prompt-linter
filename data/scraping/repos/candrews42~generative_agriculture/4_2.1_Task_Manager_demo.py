# Import required libraries
import os
import utils
import streamlit as st
import requests
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
st.set_page_config(page_title="Task Manager", page_icon="🌱") #, layout="wide")
st.header("Task Manager Chatbot")
st.write("""
**Query our actual farm task list** through natural language.

**Examples of Questions You Can Ask:**
- "What are our gardener Vinu's active tasks?"
- "What are some tasks I could complete in 30 minutes or less?"
- "What categories of tasks do we have?"
""")

# Define the main class for the Generative Agriculture Chatbot
class GenerativeAgriculture:
    # Initialize chatbot settings and API keys
    def __init__(self):
        utils.configure_openai_api_key()
        #self.openai_model = "gpt-3.5-turbo-instruct"
        #self.openai_model = "gpt-3.5-turbo"
        self.openai_model = "gpt-4-0613"
        #self.openai_model = "gpt-4-32k" # 4x context length of gpt-4
        self.notion_api_key = st.secrets["notion_api_key"]

    # Function to fetch data from Notion Database
    def responseNotion(self, readUrl):
        self.headers = {
            "Authorization": "Bearer " + self.notion_api_key,
            "Content-Type": "application/json",
            "Notion-Version": "2022-02-22"
        }
        print("here")
        res = requests.request("GET", readUrl, headers=self.headers)
        print("\ncollected data:")
        print(res)
        print(f"API Response: {res.json()}")
        
        return res.json()
    
    # Function to fetch child blocks
    def fetch_child_blocks(self, block_id):
        readUrl = f"https://api.notion.com/v1/blocks/{block_id}/children"
        res = requests.request("GET", readUrl, headers=self.headers)
        return res.json()

    # Function to convert Notion data to Markdown
    def notion_to_markdown(self, notion_data):
        print("Entering notion_to_markdown")
        markdown_output = ""
        print("\n\nnotion data")
        print(notion_data)
        # Extract and format the title
        title_list = notion_data.get("properties", {}).get("Title", {}).get("title", [])
        print(f"Title List: {title_list}")  # Debugging line
        title = title_list[0].get("plain_text", "") if title_list else ""
        markdown_output += f"# {title}\n\n"
        
        # Fetch child blocks
        block_id = notion_data.get("id")
        child_blocks = self.fetch_child_blocks(block_id).get('results', [])
        
        # Format child blocks
        for block in child_blocks:
            block_type = block.get("type")
            text_list = block.get(block_type, {}).get("rich_text", [])  # Change here
            text_content = text_list[0].get("plain_text", "") if text_list else ""
            
            if block_type == "paragraph":
                markdown_output += f"{text_content}\n\n"
            elif block_type == "heading_1":
                markdown_output += f"# {text_content}\n\n"
            elif block_type == "heading_2":
                markdown_output += f"## {text_content}\n\n"
            elif block_type == "heading_3":
                markdown_output += f"### {text_content}\n\n"
            elif block_type == "to_do":
                checked = block.get("to_do", {}).get("checked", False)
                checkbox = "[x]" if checked else "[ ]"
                markdown_output += f"- {checkbox} {text_content}\n"
            # Handle more types as needed
        print("exiting notion to markdown")
        return markdown_output

    # Function to fetch and format the task list from Notion
    def fetch_and_format_task_list_from_notion(self):
        pageID = "8fd4e70f758e4f3f88e5938cb2c0538c"
        pagereadUrl = f"https://api.notion.com/v1/pages/{pageID}"
        page_data = self.responseNotion(pagereadUrl)
        return self.notion_to_markdown(page_data)    


    # Setup database and agent chain
    @st.cache_resource
    def setup_chain(_self, chatbot_instructions):
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
        chatbot_memory = None
        # sqlagent_memory = ConversationBufferMemory()

        # Setup Chatbot
        chatbot_prompt_template = PromptTemplate(
            input_variables = ['task_list', 'user_input'],
            template=chatbot_instructions
        )
        llm=OpenAI(model_name=_self.openai_model, temperature=0.1, streaming=True)
        chatbot_agent = LLMChain(
            llm=llm, 
            memory=chatbot_memory, 
            prompt=chatbot_prompt_template, 
            verbose=True)
        
        # # Setup SQL toolkit and agent
        # toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))
        # sql_agent = create_sql_agent(
        #     llm=OpenAI(temperature=0.1, streaming=True),
        #     toolkit=toolkit,
        #     verbose=True,
        #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        # )
        return chatbot_agent
    
    # Main function to handle user input and chatbot response
    @utils.enable_chat_history
    def main(self):
        chatbot_instructions = """
            You are a helpful farm management assistant, tasked with managing a list of tasks. Here is the task list:

            {task_list}

            Your responsibilities include:
            - Filtering tasks based on user queries.

            Task Properties:
            - Assignees: (e.g. Vinu, Teaching Team, Colin, Reem, other names). default Unassigned
            - Priority: (Low, Medium, High) default Low
            - Date Due: "today", "this week", or a specific date as yyyy-mmm-dd. default None

            Guidelines:
            - USE THE EXACT FORMAT from the Formatting Example AND CATEGORIES from the task list for consistency, provide it in Markdown.
            - Keep the task list SORTED BY PRIORITY within each category.
            - Respond with ONLY THE FORMATTED TASK LIST
            - Only provide Properties if they are unique between tasks, i.e. leave out the name if I ask for a specific name's assigned tasks.

            Formatting Example:
            🌱 Planting and Harvesting
            [ ] Plant all seedlings [High]
            [ ] Send list of seeds [Medium, 25/Oct/2023]
            
            User Query:
            "{user_input}"
            """
        chatbot_agent = self.setup_chain(chatbot_instructions)
        user_query = st.chat_input(placeholder="Enter your query to filter tasks")
        
        # Assume task_markdown_content contains the markdown-formatted task list
        task_markdown_content = self.fetch_and_format_task_list_from_notion()
        
        if user_query:
            utils.display_msg(user_query, 'user')
            
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                chatbot_response = chatbot_agent.run(
                    {
                        'task_list': task_markdown_content,
                        'user_input': user_query
                    },
                    callbacks=[st_cb]
                )
                st.write(chatbot_response)

# Entry point of the application
if __name__ == "__main__":
    obj = GenerativeAgriculture()
    obj.main()