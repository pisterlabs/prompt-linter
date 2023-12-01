import os
import openai
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

import streamlit as st

API_KEY = open('key.txt', 'r').read().strip()
DB_PASSWORD = open('pass.txt', 'r').read().strip()
os.environ["OPENAI_API_KEY"] = API_KEY

openai.api_key = API_KEY

from dotenv import load_dotenv
load_dotenv()


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


def get_response(input_text):
    response = agent_executor(input_text)

    # print(response['intermediate_steps'][1][0].tool)
    # print(response['intermediate_steps'][-1][0].tool)
    # print(response['output'])


    if response['intermediate_steps'][1][0].tool == 'sql_db_schema':
        schema = response['intermediate_steps'][1][1]
    else: schema = None

    if response['intermediate_steps'][-1][0].tool == 'sql_db_query':
        query = response['intermediate_steps'][-1][0].tool_input
        query_output = response['intermediate_steps'][-1][1]
    else: query, query_output = None, None

    answer = response['output']

    return schema, query, query_output, answer

def explain(query, schema, query_output):

    message_history = [{"role": "user", "content": f"""You are a SQL query explainer bot. That means you will explain the logic of a SQL query. 
                    There is a postgreSQL database table with the following table:

                    {schema}                   
                    
                    A SQL query is executed on the table and it returns the following result:

                    {query_output}

                    I will give you the SQL query executed to get the result and you will explain the logic executed in the query.
                    Make the explanation brief and simple. It will be used as the explanation of the results. Do not mention the query itself.
                    No need to explain the total query. Just explain the logic of the query.
                    Reply only with the explaination to further input. If you understand, say OK."""},
                   {"role": "assistant", "content": f"OK"}]

    message_history.append({"role": "user", "content": query})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )

    explaination = completion.choices[0].message.content
    return explaination


host="ep-wispy-forest-393400.ap-southeast-1.aws.neon.tech"
port="5432"
database="accountsDB"
username="db_user"
password=DB_PASSWORD

# # Create the sidebar for DB connection parameters
# st.sidebar.header("Connect Your Database")
# host = st.sidebar.text_input("Host", value=host)
# port = st.sidebar.text_input("Port", value=port)
# username = st.sidebar.text_input("Username", value=username)
# password = st.sidebar.text_input("Password", value=password)
# database = st.sidebar.text_input("Database", value=database)
# # submit_button = st.sidebar.checkbox("Connect")

db = SQLDatabase.from_uri(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prefix=SQL_PREFIX,
    suffix=SQL_FUNCTIONS_SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    agent_executor_kwargs = {'return_intermediate_steps': True}
)

# Create the main panel
st.title("connectDB :star2:")
st.subheader("You are connected to AD-IQ Accounts database!!")
st.caption("The database contains the Daily Cash Input Output data for AD-IQ Accounts from Jan to June")


with st.expander("Database properties"):

    # st.divider()

    # st.write("*--Helpful Info--*")
    st.subheader("Cost categories:")
    st.text("""
    "Fixed Asset"
    "Field Asset"
    "Receipt"
    "Operating Expense"
    "Administrative Expense"
    "None"
    """)

    st.subheader("Cost Subcategories:")
    st.text("""
    "Computers & Printers"
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
    "Router"

    """)

    st.subheader("List of Holder/Bearer/Vendor:")
    st.text("""
    "Electric Spare Tools"
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
    "Router"
    """)
    # st.divider()

with st.expander("FAQs"):
    st.text("""
    1. Describe the database.
    2. What is the timeline of the data present?
    3. Who are the top 5 most expensive vendors?
    4. What is the total amount of money spent on 'Electrician'?
    5. How many different cost categories are there?
    6. What is the total ISP cost in May?
    7. Would you happen to have any information about the CEO?
    8. Give me all expenses regarding Rubab?
    9. Do we have any scope to reduce the expenses on operations?
    """)

# Get the user's natural question input
question = st.text_input(":blue[Ask a question:]", placeholder="Enter your question.")

# Create a submit button for executing the query
query_button = st.button("Submit")

# Execute the query when the submit button is clicked
if query_button:

    # Display the results as a dataframe
    # Execute the query and get the results as a dataframe
    try:
        with st.spinner('Calculating...'):
            print("\nQuestion: " + str(question))
            # print(str(question))
            schema, query, query_output, answer = get_response(question)

            if query:
                explaination = explain(query, schema, query_output)
            else: explaination = None

            # explaination = explain(query, schema, query_output)

        # if query:
        #     print("\nExplaination: " + str(explaination))

        print("\nExplaination: " + str(explaination))

        st.subheader("Answer :robot_face:")
        st.write(answer)

        try:
            if query:

                st.divider()
                # st.caption("Query:")
                # st.caption(query)

                st.caption("Explaination:")
                st.caption(explaination)

                st.divider()
        except Exception as e:
            print(e)

        st.info(":coffee: _Did that answer your question? If not, try to be more specific._")
    except:
        st.warning(":wave: Please enter a valid question. Try to be as specific as possible.")
