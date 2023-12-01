import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from components import file_container
from config import *
from debuggpt import gpt_debug
import environ
import os 
import openai




def viz():
    st.header("📊 Monitor your data quality KPI")
    ## PROMPT 
    system_prompt = """Assistant is the lead data steward in a company. He monitors the overall data quality performance for the company and can identify 
    the priority of the data issues, and can provide data quality improvement guidance to data owners."""
    instruction = """I will provide you with some data along with some null value summary statistics. 
    I want you to give me a high level interpretation of the data quality of the data I provided, and point out what should be the prority of the data issues
    Some fields in the data sets are critical and can't be Null, for example, in the data, primary keys can't be left blank, while fields that are not primary keys can be left blank.
    Focus on the Null value issue first, then look at other data quality issue such as format of the data.
    Please bucket the action recommendations into "High", "Medium", and "Low" priority."""
    
    data = file_container()
    if data is not None and len(data) > 0:
        # Create an empty DataFrame to store the null value percentages
        data_prompt = ""
        count = 1
        for file in data:
            file_string = file.to_string()
            data_prompt = data_prompt + f'Data set {count} is: ' + file_string + f'That is all for Data set {count}.'
            count += 1
        null_percentage_combined = pd.DataFrame()
        null_count_combined = pd.DataFrame()
        # Iterate through the DataFrames and calculate the null value percentage for each
        for idx, df in enumerate(data, 1):
            null_percentage = df.isnull().mean() * 100
            null_percentage_combined[f'Data Source {idx}'] = null_percentage
            null_count = df.isnull().sum()
            null_count_combined[f'Data Source {idx}'] = null_count
        # Reshape the DataFrame for Seaborn
        null_percentage_melted = null_percentage_combined.reset_index().melt(id_vars='index', var_name='Data Source', value_name='Percentage')
        null_count_melted = null_count_combined.reset_index().melt(id_vars='index', var_name='Data Source', value_name='Count')
        null_percentage_melted_string = null_percentage_melted.to_string(index=False)
        null_count_string = null_count_combined.to_string()
        # Plotting the bar chart with Seaborn
        plt.figure(figsize=[8,4])
        fig = sns.barplot(x='index', y='Percentage', hue='Data Source', data=null_percentage_melted)
        plt.title('Percentage of Null Values in Each Column Across Data Sources')
        plt.ylabel('Percentage (%)')
        plt.xlabel('Columns')
        # Rotate the x-axis labels
        plt.xticks(rotation=45)
        plt.legend(title='Data Source')
        for p in fig.patches:
            fig.annotate(format(p.get_height(), '.1f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points', fontsize=9)
        st.pyplot(plt)

            ## DISPLAY RESPONSE 
        if st.button("Check Data Quality KPI Health", type="primary"):
            response = llm([
                SystemMessage(content=system_prompt),
                HumanMessage(content=instruction + "Here's the Null value percentage distribution for each data set: " + null_percentage_melted_string)
            ]).content
            # llm = openai.ChatCompletion.create(**parameters)
            # llm_dict = dict(llm)
            # content = llm_dict['choices'][0]['message']['content']
            st.markdown(response)






