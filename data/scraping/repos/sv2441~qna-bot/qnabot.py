import streamlit as st
import base64
import csv
import os
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize chat model
model_option = st.checkbox("Select Model: GPT 3.5 Turbo",)
if model_option:
    model = "gpt-3.5-turbo"
    output_filename = "gpt3.5_results.csv"  # Set the output filename for GPT 3.5 Turbo
else:
    model = "gpt-4"
    output_filename = "gpt4_results.csv"  # Set the output filename for GPT 4

# Initialize chat model
chat_llm = ChatOpenAI(model_name=model, temperature=0.0)

def dict_to_csv(data_list):
    with open('data11.csv', 'w', newline='') as csvfile:
        fieldnames = ['Correct Answer', 'Explanation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if the file is empty and write the header only if it's empty
        is_file_empty = csvfile.tell() == 0
        if is_file_empty:
            writer.writeheader()

        for data in data_list:
            writer.writerow(data)

def generator(df, output_filename):
    answer_schema = ResponseSchema(name="Correct Answer", description="Correct answer Option.")
    explanation_schema = ResponseSchema(name="Explanation", description="Explanation in one Sentence.")

    response_schemas = [answer_schema, explanation_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    title_template = """
                    for the "{question}", Provide the correct answer option (A, B, C, D).
                    If none of the options apply, please select "N/A".
                    Provide Explanation in one Sentence.
                    {format_instructions}
                    """ 
    prompt = ChatPromptTemplate.from_template(template=title_template)
    
    data_list = []  # Create a list to store dictionaries

    for index, row in df.iterrows():
        messages = prompt.format_messages(question=row['Question'], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        data_list.append(data)  # Append the data to the list
    
    dict_to_csv(data_list)  # Pass the list of dictionaries to dict_to_csv
    
    data11 = pd.read_csv("data11.csv", encoding='cp1252')
    results = pd.concat([df, data11], axis=1)
    results.to_csv(output_filename, mode='a', header=not os.path.isfile(output_filename), index=False)
    st.subheader("Results")
    st.dataframe(results)
    st.markdown(get_download_link(results, output_filename), unsafe_allow_html=True)

def get_download_link(df, output_filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{output_filename}">Download CSV File</a>'
    return href

def main():
    st.title("👨‍💻 Question and Answer")
    # File upload
    file = st.file_uploader("Upload OP's", type=["csv"])

    if file is not None:
        # Read CSV file
        df = pd.read_csv(file)

        # Display preview
        st.subheader("CSV File Preview")
        st.dataframe(df)

        # Button to process the file
        if st.button("Generate answers"):
            generator(df, output_filename)
            
if __name__ == "__main__":
    main()
