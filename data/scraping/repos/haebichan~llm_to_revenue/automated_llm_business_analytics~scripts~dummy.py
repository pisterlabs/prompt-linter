import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import openai
import os
import requests
import random
import io
import base64
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

from IPython.display import display, clear_output, HTML
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

CHROMA_PATH = "./chroma_db"
GPT_MODEL = "gpt-3.5-turbo"
SUPPORT_DOC_FOLDER_PATH = "support_docs"
SCORE_THRESHOLD = 0.65


# Set the page configuration to landscape mode
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Generate a list of departments
departments = ['HR', 'Finance', 'Marketing', 'Sales', 'Engineering']

# Create an empty list to store data
data = []

# Generate dummy data
for _ in range(100):
    month = random.choice(['January', 'February', 'March'])
    department = random.choice(departments)
    employee_name = f'Employee{_ + 1}'
    spending_amount = round(random.uniform(1000, 5000), 2)
    # Round the salary to the nearest thousand
    salary = round(random.uniform(30000, 80000) / 1000) * 1000
    
    
    # Check for duplicate employee names in different departments
    while any((d['Employee Name'] == employee_name) and (d['Department'] != department) for d in data):
        employee_name = f'Employee{_ + 1}'
    
    data.append({
        'Month': month,
        'Department': department,
        'Employee Name': employee_name,
        'Spending Amount': spending_amount,
        'Salary': salary  # Add the 'Salary' column
    })

# Create a DataFrame
df = pd.DataFrame(data)


# Replace with your OpenAI API key
api_key = os.environ.get('OPENAI_API_KEY')

# Initialize the OpenAI API client
openai.api_key = api_key


def get_completion(prompt, model):

    messages = [{"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
    
    model=model,
    
    messages=messages,
    
    temperature=0
    )
    return response


def sample_df_with_fully_unique_row_across_all_columns(df):
    # Initialize an empty DataFrame to store the sampled rows
    sampled_df = pd.DataFrame(columns=df.columns)
    
    # Initialize a set to keep track of sampled rows as tuples
    sampled_rows = set()
    
    # Loop through the rows and add them to the sampled_df while checking for uniqueness
    for index, row in df.iterrows():
        row_tuple = tuple(row)
        if row_tuple not in sampled_rows:
            sampled_df = pd.concat([sampled_df, row.to_frame().T])
            sampled_rows.add(row_tuple)
    
        # Check if you have enough unique rows; you can adjust the number as needed
        if len(sampled_df) >= 5:
            break
    
    return sampled_df

################################################################################################################################


# Define the dynamic code as a string
sampled_data = sample_df_with_fully_unique_row_across_all_columns(df)

# Create a list to store generated charts
generated_charts = []


def process_doc_files(SUPPORT_DOC_FOLDER_PATH) -> dict:
    """
    Process text files in a folder and return a dictionary with file names as keys and content as values.

    :param folder_path: The path to the folder containing the text files.
    :return: A dictionary with file names as keys and content as values.
    """
    # Initialize an empty dictionary to store the results
    doc_dict = {}

    # Check if the folder exists
    if not os.path.exists(SUPPORT_DOC_FOLDER_PATH):
        return doc_dict  # Return an empty dictionary if the folder does not exist

    # List all files in the folder
    file_list = os.listdir(SUPPORT_DOC_FOLDER_PATH)

    # Iterate through the files
    for filename in file_list:
        # Check if the file has a .txt extension
        if filename.endswith(".txt"):
            # Create the full path to the file
            file_path = os.path.join(SUPPORT_DOC_FOLDER_PATH, filename)

            # Open the file and read its content
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            # Store the content in the dictionary with the filename as the key
            doc_dict[filename] = file_content

    return doc_dict


# Define the function to extract the "document_objects" object
def extract_document_objects(all_file_mapping):

    document_objects = []

    for title, doc in all_file_mapping.items():
        
        document_object = Document(page_content= doc)
        document_objects.append(document_object)    

    return document_objects

def folder_exists(folder_path):
    return os.path.exists(folder_path) and os.path.isdir(folder_path)


all_docs = process_doc_files(SUPPORT_DOC_FOLDER_PATH)

document_objects = extract_document_objects(all_docs)

if folder_exists(folder_path = CHROMA_PATH):
    # Get the vectorstore saved previously
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
else:
    # Create new vectorstore
    vectorstore = Chroma.from_documents(documents=document_objects, embedding=OpenAIEmbeddings(), persist_directory = CHROMA_PATH)
    
# Create a Streamlit web app
st.title('LLM Analytics Dashboard')

# Initialize a session state variable to store user requests
if 'user_requests' not in st.session_state:
    st.session_state.user_requests = []

# Add a text input for user-defined code
user_query = st.text_area('Type Request for Visual Chart Creation:', height=25)

# Add a button to save the user request
if st.button('Save Request'):
    if user_query:
        st.session_state.user_requests.append(user_query)
        st.success('Request saved!')
        st.text("User Requests:")
        for i, request in enumerate(st.session_state.user_requests):
            st.text(f"{i + 1}: {request}")
        # Clear the text input
        user_query = ""

# Add a button to create the visual
if st.button('Create Visual'):
    if st.session_state.user_requests:
        st.text("Processing User Requests:")
        num_requests = len(st.session_state.user_requests)
        num_cols = min(2, num_requests)  # Set the maximum number of columns to 2
        
        # Calculate the width of each chart based on the number of columns
        chart_width = 100 / num_cols
        
        for i, request in enumerate(st.session_state.user_requests):            

           retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": SCORE_THRESHOLD, 'k':1})
           # Get the retrieved_docs
           retrieved_doc_list = retriever.get_relevant_documents(request)
           if len(retrieved_doc_list) == 0:
               code_assistance_doc = ''
           else:
               code_assistance_doc = retrieved_doc_list[0].page_content
    
           prompt = f"""Generate code for following task: "Create an appropriate visual to show {user_query} using matplotlib"
    
               Sample Data:
               {sampled_data}
               
               {code_assistance_doc}
               
               Instructions for Code Generation:
               1. Assume data is stored in a DataFrame called "df". Don't create any dummy data on your own.
               2. Please only provide just the code as the output."""
           res = get_completion(prompt=prompt, model = GPT_MODEL).choices[0].message['content']
           
           try: 
               exec(res)
               
               # Create a placeholder for the chart with dynamic width
               chart_placeholder = f'<div style="width: {chart_width}%; display: inline-block;">'
               chart_placeholder += f'<h3>Chart {i + 1}</h3>'
               
               # Convert the Matplotlib figure to a base64-encoded image
               buf = io.BytesIO()
               plt.savefig(buf, format='png')
               buf.seek(0)
               chart_base64 = base64.b64encode(buf.read()).decode()
               buf.close()
               
               chart_placeholder += f'<img src="data:image/png;base64,{chart_base64}">'
               chart_placeholder += '</div>'
               
               generated_charts.append(chart_placeholder)
               
           except Exception as e:
               st.write(user_query, ' not accepted. Please try again')
               st.write('\nLLM output', res)
            
        # Combine all generated charts into a single HTML element
        chart_html = ''.join(generated_charts)
        st.markdown(chart_html, unsafe_allow_html=True)
            
    else:
        st.warning("No user requests to process. Please save requests first.")

# Display the DataFrame at the bottom of the dashboard
st.markdown("<br><br>", unsafe_allow_html=True)  # Add multiple blank lines for spacing
st.text("Sample Source Data (DataFrame):")
st.dataframe(sampled_data)
st.text("")  # Empty text to add spacing
