import streamlit as st
import pandas as pd
import random
import openai
import os
import matplotlib.pyplot as plt
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma # vector database

CHROMA_PATH = './chroma_db'
SUPPORT_DOC_FOLDER_PATH = 'support_docs'

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


def folder_exists(folder_path):
    return os.path.exists(folder_path) and os.path.isdir(folder_path)

# Define the function to extract the "document_objects" object
def extract_document_objects(all_file_mapping):

    document_objects = []

    for title, doc in all_file_mapping.items():
        
        document_object = Document(page_content= doc)
        document_objects.append(document_object)    

    return document_objects

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


# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key = os.environ['OPENAI_API_KEY']

# Initialize the OpenAI API client
openai.api_key = api_key

def generate_text(prompt, temperature=0.1):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can choose an appropriate engine
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)



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
    
    # Check for duplicate employee names in different departments
    while any((d['Employee Name'] == employee_name) and (d['Department'] != department) for d in data):
        employee_name = f'Employee{_ + 1}'
    
    data.append({
        'Month': month,
        'Department': department,
        'Employee Name': employee_name,
        'Spending Amount': spending_amount
    })

# Create a DataFrame
df = pd.DataFrame(data)


#####


sampled_data = sample_df_with_fully_unique_row_across_all_columns(df)

all_docs = process_doc_files(SUPPORT_DOC_FOLDER_PATH)

document_objects = extract_document_objects(all_docs)


if folder_exists(folder_path = CHROMA_PATH):
    vectorstore = Chroma(persist_directory = CHROMA_PATH, embedding_function = OpenAIEmbeddings())
else:

    vectorstore = Chroma.from_documents(documents = document_objects, embedding = OpenAIEmbeddings(), persist_directory = CHROMA_PATH)

if 'user_requests' not in st.session_state:
    st.session_state.user_requests = []

st.title("LLM Analytics Dashboard")

user_query = st.text_area("Type Request for Visual Chart Creation: ", height = 25)

if st.button('Save Request'):
    if user_query:
        st.session_state.user_requests.append(user_query)
    
        st.success("Request Saved!")

        for i, request in enumerate(st.session_state.user_requests):
            st.text(f"{i+1}: {request}")

num_columns = st.slider("Select the number of columns to display on dashboard", 1, 5, 3)

if st.button('Execute Request'):
    if st.session_state.user_requests:

        figures = []

        for index, user_query in enumerate(st.session_state.user_requests):

            retriever = vectorstore.as_retriever(serarch_type = "similarity_score_threshold", search_kwargs = {"score_threshold": 0.65, 'k': 1})
            retrieved_doc_list = retriever.get_relevant_documents(user_query)
    
            if len(retrieved_doc_list) == 0:
                code_assistance_doc = " "
    
            else:
                code_assistance_doc = retrieved_doc_list[0].page_content
    
            prompt = f"""Generate code for this request "Create an appropriate visual to show {user_query}"
    
            Sample Data:
            {sampled_data}  # Define or replace sampled_data with your actual data
            
            {code_assistance_doc}
            
            Instructions for Code Generation:
            1. Write code to filter the data for the specified column and values when necessary.
            2. Assume data is just called "df". Don't create any dummy data on your own
            3. Use matplotlib (plt) to generate visuals
            4. Just give me the code as output"""
    
            
            generated_ai_output = generate_text(prompt = prompt)
    
            exec_globals = {'df':df}
        
            exec(generated_ai_output, exec_globals)
    
            if 'plt' in exec_globals:
                figure = exec_globals['plt'].gcf()

                figures.append(figure)
                plt.close(figure)
                # st.pyplot(figure)

    for i in range(0, len(figures), num_columns):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            if i + j < len(figures):
                cols[j].pyplot(figures[i+j])

st.write(sampled_data)










