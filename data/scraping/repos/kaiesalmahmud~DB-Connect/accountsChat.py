import streamlit as st
from pathlib import Path
# from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

st.set_page_config(page_title="accountsChat", page_icon="🦜")
st.title("🦜 accountsChat")

# # User inputs
# radio_opt = ["Use sample database - Chinook.db", "Connect to your SQL database"]
# selected_opt = st.sidebar.radio(label="Choose suitable option", options=radio_opt)
# if radio_opt.index(selected_opt) == 1:
#     db_uri = st.sidebar.text_input(
#         label="Database URI", placeholder="mysql://user:pass@hostname:port/db"
#     )
# else:
#     db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
#     db_uri = f"sqlite:////{db_filepath}"

import os
openai_api_key = open('key.txt', 'r').read().strip()
os.environ["OPENAI_API_KEY"] = openai_api_key
DB_PASSWORD = open('pass.txt', 'r').read().strip()

from dotenv import load_dotenv
load_dotenv()

host="ep-wispy-forest-393400.ap-southeast-1.aws.neon.tech"
port="5432"
database="accountsDB"
username="db_user"
password=DB_PASSWORD

db_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

# openai_api_key = st.sidebar.text_input(
#     label="OpenAI API Key",
#     type="password",
# )

# Check user inputs
if not db_uri:
    st.info("Please enter database URI to connect to your database.")
    st.stop()

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Setup agent
# llm = OpenAI(openai_api_key=openai_api_key, temperature=0, streaming=True)
llm = ChatOpenAI(model_name="gpt-4", temperature=0)


@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri)


db = configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Following are the unique values in some of the columns of the database. Search for these values in their corresponding columns to get the relevant information:

Unique values in column 'cost_category': ["Fixed Asset"
"Field Asset"
"Receipt"
"Operating Expense"
"Administrative Expense"
"None"]


Unique values in column 'cost_subcategory': ["Computers & Printers"
"Software & Subscriptions"
"Furniture & Fixtures"
"None"
"Pantry Supplies"
"Office Stationary"
"Travelling & Conveyance"
"Misc. Exp"
"ISP"
"City Group Accounts"
"Electrician - Tv Installation"
"Stata IT Limited"
"Sheba.xyz- digiGO"
"Salary (Op)"
"Advertising"
"Hasan & Brothers"
"CEO"
"KPI Bonus"
"Final Settlement"
"Software & Subscription Fees"
"Electric Equipment"
"IOU"
"Medicine"
"Training & Development"
"Sales"
"Bill Reimbursement"
"Lunch Allowance"
"Balance B/D"
"Deployment Equipments "
"Retail Partner"
"Electric Tools - Tv Installation"
"Office Decoration/Reconstruction"
"Entertainment (Ops)"
"Carrying Cost"
"Entertainment (Admin)"
"Festival Bonus"
"Office Refreshment"
"Office Equipment"
"Bkash"
"Router"]


Unique values in column 'holder_bearer_vendor_quantity_device_name': ["Electric Spare Tools"
"75"
"None"
"Salim"
"Rakibul"
"Shoikot"
"Morshed"
"Android Box Tx6"
"ISP"
"Tv Frame"
"25"
"Hasan & Brothers"
"Digi Jadoo Broadband Ltd"
"H & H Construction"
"Teamviewer"
"Tea Spices"
"Amzad"
"Vendor"
"100"
"Omran"
"Flash Net Enterprise"
"Grid Ventures Ltd"
"32 Tv"
"Aman"
"Retail Partner"
"Printer"
"Shahin"
"Umbrella"
"Masud"
"A/C Payable"
"Tea"
"Coffee"
"Staffs"
"Emon"
"Flat flexible cable"
"May"
"Working Capital"
"Eid-ul-fitre"
"Shamim"
"Rubab"
"SR"
"CEO"
"WC"
"SSD 256 GB"
"Accounts (AD-IQ)"
"Retail Partner's Payment"
"Condensed Milk"
"Electrician"
"Farib & Indec"
"Jun"
"Asif"
"Driver"
"Nut+Boltu"
"Sugar"
"Labib"
"April"
"Coffee Mate"
"Tonner Cartridge"
"Router"]


Unique values in column 'source': ["50K" 
"SR" 
"None"]

Following are some exmaple question and their corresponding queries:

Question: give me top 10 cash out in may?
Query: SELECT date, details, cash_out FROM ledger WHERE EXTRACT(MONTH FROM date) = 5 AND cash_out IS NOT NULL  ORDER BY cash_out DESC LIMIT 10;
Observation: When ordering by a column in descending order, the top values will be the largest values in the column.

"""

SQL_FUNCTIONS_SUFFIX = """I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables."""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

# agent = create_sql_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prefix=SQL_PREFIX,
    suffix=SQL_FUNCTIONS_SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    # agent_executor_kwargs = {'return_intermediate_steps': True}
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "You are connected with accountsDB. Ask questions!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)