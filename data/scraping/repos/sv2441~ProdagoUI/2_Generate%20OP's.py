## import the CSV file and output the CSV file 

import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
import csv
import os
import base64


load_dotenv()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
    
chat_llm = ChatOpenAI(temperature=0.0)


def dict_to_csv(data, filename, append=False):
    mode = 'a' if append else 'w'
    with open(filename, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not append:
            writer.writeheader()
        writer.writerow(data)

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="final.csv">Download CSV File</a>'
    return href

def convert_dict_to_csv(data_dict):
    with open('data11.csv', 'w', newline='') as csvfile:
        fieldnames = ['OP type', 'Activity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if the file is empty and write the header only if it's empty
        is_file_empty = csvfile.tell() == 0
        if is_file_empty:
            writer.writeheader()

        for key, value in data_dict.items():
            if isinstance(value, list):
                for item in value:
                    writer.writerow({'OP type': key, 'Activity': item})
            else:
                writer.writerow({'OP type': key, 'Activity': value})



def result(df):
    
    #####output parser #############################################

    Action_schema = ResponseSchema(name="Action",
                             description="List broad actionables relating to requirements. Classify each actionable into Requirements or Support practices.")

    response_schemas = [ Action_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    ###########################################################################

    title_template = """ \ You are an AI Governance bot. Just Execute the set of steps.
                1)provide List broad actionables relating to "{topic}".Classify each actionable into Requirements or Support practices. 
                If the "{topic}" demands some actions, that is referred as 'Requirement'. 
                If the "{topic}" expects an outcome, the actionables that will help achieving that is referred as 'Support Practices'.
                Ensure that the actionables are activites and Ensure each point has an action word, the subject and the activity.
               {format_instructions}
                """


    prompt = ChatPromptTemplate.from_template(template=title_template)

    df=df
    for index, row in df.iterrows():
        messages = prompt.format_messages(topic=row['Requirements'], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        convert_dict_to_csv(data)
        
        test=pd.DataFrame(df.iloc[index]).T
        data11 = pd.read_csv(r'data11.csv',encoding='cp1252')
        results = pd.concat([test, data11], axis=1).fillna(0)
        # result.to_csv('final1.csv')
        test=pd.DataFrame(df.iloc[0]).T
        results = pd.concat([test, data11], axis=1).fillna(0)
        results.to_csv('results.csv', mode='a', header=not os.path.isfile('results.csv'), index=False)

    data11 = pd.read_csv('data11.csv')
    # final = pd.read_csv('final.csv')
    st.subheader("Final Result")
    st.dataframe(data11)
    st.markdown(get_download_link(data11), unsafe_allow_html=True)


def results2(df):
    summary_schema = ResponseSchema(name="Summary",
                             description="Summary of Action Associated in Text in 15 to 20 words.")

    response_schemas = [summary_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    title_template = """ \ You are an AI Governance bot.
                Summarize actions associated with the following statement in 15-20 words paragraphin "{topic}"
                {format_instructions}
                """

    prompt = ChatPromptTemplate.from_template(template=title_template)
    # df=pd.read_csv("final1.csv")
    df2=df['Activity']
    for i in range(len(df2)):
        messages = prompt.format_messages(topic=df2[i], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        dict_to_csv(data, 'data12.csv', append=True)
    data12 = pd.read_csv('data12.csv', names=['Description'])
    result = pd.concat([df, data12], axis=1)
    output= pd.concat([df2, data12], axis=1)
    result.to_csv('final12.csv')
    data12.to_csv('data12.csv')
    final_result=pd.read_csv('results.csv')
    results = pd.concat([final_result, data12], axis=1)
    results.to_csv('results.csv')
    st.subheader("Summary Result")
    st.dataframe(output)
    st.markdown(get_download_link(results), unsafe_allow_html=True)
    

def results4(df):
    Artefact_basis_schema = ResponseSchema(name="Artefact Name",
                                description="Provide a name for the artefact basis")

    response_schemas = [Artefact_basis_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    title_template = """ \ You are an AI Governance bot.
                Provide a name for the artefact basis the following text “{topic}”.
                {format_instructions}
                """

    prompt = ChatPromptTemplate.from_template(template=title_template)
    # df=pd.read_csv("final1.csv")
    df2=df['Description']
    for i in range(len(df2)):
        messages = prompt.format_messages(topic=df2[i], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        dict_to_csv(data, 'data14.csv', append=True)
    data14 = pd.read_csv('data14.csv', names=['Artefact Name'])
    result = pd.concat([df, data14], axis=1)
    output= pd.concat([df2, data14], axis=1)
    result.to_csv('final14.csv')
    data14.to_csv('data14.csv')
    final_result=pd.read_csv('results.csv')
    results = pd.concat([final_result, data14], axis=1)
    results.to_csv('results.csv')
    st.subheader("Artefact Result")
    st.dataframe(output)
    st.markdown(get_download_link(results), unsafe_allow_html=True)
    
    
def results5(df):
    Artefact_description_schema = ResponseSchema(name="Artefact Description",
                             description="Provide an artefact description.")

    response_schemas = [Artefact_description_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    title_template = """ \ You are an AI Governance bot.
                Provide an artefact description based on the following “{topic}”.
                {format_instructions}
                """

    prompt = ChatPromptTemplate.from_template(template=title_template)
    # df=pd.read_csv("final1.csv")
    df2=df['Artefact Name']
    
    for i in range(len(df2)):
        messages = prompt.format_messages(topic=df2[i], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        dict_to_csv(data, 'data15.csv', append=True)
    data15 = pd.read_csv('data15.csv', names=['Artefact Description'])
    result = pd.concat([df, data15], axis=1)
    output= pd.concat([df2, data15], axis=1)
    result.to_csv('final15.csv')
    data15.to_csv('data15.csv')
    final_result=pd.read_csv('results.csv')
    results = pd.concat([final_result, data15], axis=1)
    results.to_csv('results.csv')
    st.subheader("Artefact Description")
    st.dataframe(output)
    final_result=pd.read_csv('results.csv',usecols=['Requirements','OP Type','Activity','Description',	'Intended Results','Artefact Name',	'Artefact Description'])
    
    st.markdown(get_download_link(final_result), unsafe_allow_html=True)
    
    
def results3(df):
    intended_results_schema = ResponseSchema(name="Intended Results",
                             description="Summary of intended results.")

    response_schemas = [intended_results_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    
    title_template = """ \ You are an AI Governance bot.
                Summarize intended results of doing the activity from a third person perspective for “{topic}” in 15 to 20 words. 
                {format_instructions}
                """

    prompt = ChatPromptTemplate.from_template(template=title_template)
    # df=pd.read_csv("final1.csv")
    df2=df['Description']
    for i in range(len(df2)):
        messages = prompt.format_messages(topic=df2[i], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        dict_to_csv(data, 'data13.csv', append=True)
    data13 = pd.read_csv('data13.csv', names=['Intended Results'])
    result = pd.concat([df, data13], axis=1)
    output= pd.concat([df2, data13], axis=1)
    result.to_csv('final13.csv')
    data13.to_csv('data13.csv')
    final_result=pd.read_csv('results.csv')
    results = pd.concat([final_result, data13], axis=1)
    results.to_csv('results.csv')
    st.subheader("Intended Result")
    st.dataframe(output)
    st.markdown(get_download_link(results), unsafe_allow_html=True)
    


def main():
    st.image('logo.png')
    st.title("Upload Requirements")

    # File upload
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        # Read CSV file
        # df = pd.read_csv(file)
        df = pd.read_csv(file)

        # Display preview
        st.subheader("CSV File Preview")
        st.dataframe(df)

        # Button to process the file
        if st.button("Generate Activity"):
            result(df)
            
        if st.button("Generate Description"):
            results2(pd.read_csv('results.csv'))
            
        if st.button("Generate Intended Results"):
            results3(pd.read_csv('results.csv'))
            
        if st.button("Generate Artefact"):
            results4(pd.read_csv('results.csv'))
            
        if st.button("Generate Artefact Description"):
            results5(pd.read_csv('results.csv'))
            
        


if __name__ == "__main__":
    main()
