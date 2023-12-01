from dotenv import load_dotenv

load_dotenv()

import os
import streamlit as st
import pandas as pd

from langchain import PromptTemplate
from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.llms import OpenAI
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.utilities import SerpAPIWrapper

llm = OpenAI(temperature=0.9)
tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
my_name = "Nick"
my_role = "founder"

template = """
You are a cold email outreach expert named {my_name} selling {product} with the function {function}.
Your role at this company is {my_role}.
Search for a person named {prospect} and craft a cold email with 3 paragraphs that 
- contains an introduction
- describes how {product} can help them and tailor it to their role and company
- proposes to schedule a video call

Don't be too verbose with your writing. Keep the email relatively short. The goal is to get a resopnse from the prospect.

Do not label the paragraphs.
Make sure to start a new line after each paragraph.
"""

prompt = PromptTemplate(
    input_variables=["my_name", "my_role", "product", "function", "prospect"],
    template=template,
)

# Set the title of the app
st.title('CSV to cold email')

product = st.text_input('Describe your service: ') # AI agency
function = st.text_input('Describe what your product does: ') # Save time and money through automation

# Create a file uploader widget to upload the CSV file
uploaded_file = st.file_uploader('Upload a CSV file', type='csv')

if uploaded_file is not None:
    # Read the CSV file using pandas
    data = pd.read_csv(uploaded_file)

    # Display the first 5 rows of the dataset
    st.write('Displaying the first 5 rows of the uploaded CSV:')
    st.dataframe(data.head())

    data_as_dicts = data.to_dict('records')

    allowed_col_names = [
        'First Name', 
        'Last Name', 
        'Person Linkedin Url', 
        'Company Linkedin Url', 
        'Company Name', 
        'Position'
    ]

    # Display table header with column names
    header_columns = st.columns(data.shape[1] + 1)
    header_columns[0].write('')
    for i, col_name in enumerate(data.columns):
        if col_name in allowed_col_names:
            header_columns[i+1].write(col_name)

    showing_count = 5

    # Display rows with a checkbox for selection
    selected_rows = []
    selections = []
    for i, row_dict in enumerate(data_as_dicts[:showing_count]):
        row_columns = st.columns(data.shape[1] + 1)
        selection = row_columns[0].checkbox(f"", key=f"row{i+1}")
        selections.append(selection)
        
        for j, (col_name, cell_value) in enumerate(row_dict.items()):
            if col_name in allowed_col_names:
                row_columns[j+1].write(cell_value)
        
        if selection:
            selected_rows.append(row_dict)

    def write_to_file(folder, filename, text):
        # Check if the folder exists, if not, create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Write the text to the file
        with open(os.path.join(folder, filename), 'w') as f:
            f.write(text)

    def create_cold_email(row):
        first_name = row['First Name']
        last_name = row['Last Name']
        name = f"{first_name} {last_name}"
        formattedPrompt = prompt.format(
            my_name=my_name, 
            my_role=my_role,
            product=product,
            function=function,
            prospect=name
        )
        result = agent.run(formattedPrompt)
        return result

    def create_cold_emails(selected_rows):
        for row in selected_rows:
            cold_email = create_cold_email(row)
            folder = 'cold_emails'
            filename = f"{row['First Name']}_{row['Last Name']}_{row['Email']}.txt"
            write_to_file(folder, filename, cold_email)

    # Button to trigger the function with the selected rows
    if st.button('Process Selected Rows'):
        create_cold_emails(selected_rows)
        
        # Reset selections after processing
        for i in range(len(selections)):
            selections[i] = False