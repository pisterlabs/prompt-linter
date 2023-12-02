# Importing necessary libraries
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.llms import Banana
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser
import csv
import tempfile
import base64
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize AzureChatOpenAI language model with specific configurations
chat_llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    model_name="gpt-35-turbo",
    temperature=0.6
)

# Setting up multiple language models with different configurations
llm_options = {
    "gpt-35-turbo": AzureChatOpenAI(
        deployment_name="gpt-35-turbo",
        model_name="gpt-35-turbo",
        temperature=0.6  # Default temperature setting
    ),
    "mistral": Banana(
        model_key="",
        model_url_slug="demo-mistral-7b-instruct-v0-1-lnlnzqkn5a",
    ),
    "llama2-13b": Banana(
        model_key="",
        model_url_slug="llama2-13b-chat-awq-loh5cxk85a",
    )
}

# Determine the execution environment (development or production)
is_dev = os.getenv("IS_DEV", "false").lower() == "true"
data_path = "data" if is_dev else "/data"
sample_data_path = "sample_data"

# Function to retrieve a prompt for a given action from a CSV file
def get_prompt_for_action(action):
    prompts_df = pd.read_csv(os.path.join(sample_data_path, "prompts.csv"))  # Read prompts CSV
    prompt_row = prompts_df[prompts_df['Action'] == action]
    if not prompt_row.empty:
        return prompt_row['Prompt'].iloc[0]
    else:
        return None 

# Function to write a string to a CSV file, with an option to append or overwrite
def str_to_temp_csv(data, temp_file):
    writer = csv.writer(temp_file)
    writer.writerow([data])
    temp_file.flush()  # Make sure data is written
    return temp_file # Append the string as a single row

# Function to generate a download link for a DataFrame in CSV format
def get_download_link(df):
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f"Operational_Requirements_{current_datetime}.csv"
    
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download CSV File</a>'
    
    return href


# Function to generate artefact descriptions for operational requirements
def artefact_description_generator(df,llm):
    Artefact_description_schema = ResponseSchema(name="Artefact Description",
                                                 description="Provide an artefact description.")

    response_schemas = [Artefact_description_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=chat_llm)

    title_template = get_prompt_for_action('Operational Requirements Artefact Description')

    prompt = ChatPromptTemplate.from_template(template=title_template)

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='', encoding='utf-8') as temp_file:
        for index, row in df.iterrows():
            count = 0
            while count < 5:
                try:
                    messages = prompt.format_messages(topic=row["Operational Requirements Description"], format_instructions=format_instructions)
                    response = chat_llm(messages)
                    response_as_dict = new_parser.parse(response.content)
                    data = response_as_dict
                    data=data.get('Artefact Description')
                    str_to_temp_csv(data, temp_file)
                    break
                
                except Exception as e:
                    print(f"Attempt {count + 1}: Failed to process row - {e}")
                    count += 1

                    if count >= 5:
                        print("Maximum retries reached. Moving to the next row.")
                        str_to_temp_csv(data, temp_file)
                        break
        temp_file.seek(0)  # Go back to the start of the file
        data15 = pd.read_csv(temp_file.name, names=['Operational Requirements Artefact Description'])            
    result = pd.concat([df, data15], axis=1)
    result.to_csv('PA-results.csv',index=False)
    st.write("Done for Operational Requirements Artefact Description")
    st.dataframe(result)
    # st.markdown(get_download_link(result), unsafe_allow_html=True)

# Function to generate specifications for operational requirements
def specifications_generator(df,llm):
    Artefact_basis_schema = ResponseSchema(name="Specifications",
                                           description="Provide a name for Specifications")

    response_schemas = [Artefact_basis_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # List any specific conditions or considerations mentioned in the following "{topic}".

    title_template = get_prompt_for_action('Operational Requirements Specifications')

    prompt = ChatPromptTemplate.from_template(template=title_template)

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='', encoding='utf-8') as temp_file:

        for index, row in df.iterrows():
            messages = prompt.format_messages(topic=row["Operational Requirements Description"])
            response = chat_llm(messages)
            data = str(response.content)  # Convert data to string
            str_to_temp_csv(data, temp_file)
            
        temp_file.seek(0)  # Go back to the start of the file
        data16 = pd.read_csv(temp_file.name, names=['Operational Requirements Specifications'])            
    result = pd.concat([df, data16], axis=1)
    result.to_csv('PA-results.csv', index=False)
    st.subheader("Done for Operational Requirements Generation")
    st.dataframe(result)
    st.markdown(get_download_link(result), unsafe_allow_html=True)

# Main function to run the Streamlit application
def main():
    # Setting up the Streamlit interface
    st.image('logo.png')
    st.title("👨‍💻 OP Completeness 2")

    st.write("This application is a prototype for generating Operational Requirements Artefact Descriptions and Intended Results.")
    st.write("""
             Input:- A CSV file with OP Title and OP Description.\n
             Output:- Operational Requirements Artefact Description and Spefication.
               """)

    # File upload functionality
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.markdown("### Download Sample CSV")
    sample = pd.read_csv(os.path.join(sample_data_path, "OP_C2_sample.csv"))
    st.markdown(get_download_link(sample), unsafe_allow_html=True)
    
    # Dropdown to select a language model
    selected_llm = st.selectbox("Select a Language Model", options=list(llm_options.keys()))  

    # Set the language model based on user selection
    llm = llm_options[selected_llm]

    # Processing uploaded file and generating operational requirements
    if file is not None:
        L1_df = pd.read_csv(file)

        # Display CSV file preview
        st.subheader("CSV File Preview")
        st.dataframe(L1_df)

       
        if st.button("Generate"):
            artefact_description_generator(L1_df,llm)
            specifications_generator(pd.read_csv('PA-results.csv'),llm)

# Entry point of the script
if __name__ == "__main__":
    main()
