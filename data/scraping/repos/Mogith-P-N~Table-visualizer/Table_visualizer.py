# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:42:14 2023

@author: mogit
"""

import pandas as pd
import streamlit as st
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import os


openai.api_key = st.secrets["OPENAI_API_KEY"]  #API key stored in secrets of Streamlit app as environment variable.


# This function will generate the code, which in turn used to display graphs from the given data.
def generate_visualization_code(prompt):
    # Use OpenAI's API to generate Python code for visualization
    response = openai.Completion.create(
        engine='text-davinci-003', 
        prompt=prompt,
        temperature=0.75, #0.75 gives pretty much consistent values.
        max_tokens=1000,   #restricting max tokens to 600.
        n=1,
        stop=None
    )

    # Fetch the generated code from OpenAI's response
    visualization_code = response.choices[0].text.strip()

    return visualization_code

def main():
    #page setup 
    st.set_page_config(page_title="Table visualization", layout="wide")
    st.title("Table visualizer with OpenAI")
    file=st.file_uploader("Kindly upload CSV file of the data you want to visualize", type='csv') #For now App will take files in the csv format.
    if file is not None:
        file_csv=pd.read_csv(file)
        df=file_csv.sample(n=30) # To avoid exhaustion of token size, I'm restricting the model input to 30 rows (which will be of less tokens).
        # Display the uploaded DataFrame
        st.subheader("Data to be visualized")
        st.write(df)

        #prompt will fetch the code needed to display 5 charts/graphs for the given Dataset. We have instructed to return in particular format.
        prompt=f"""
        Perform the following actions:
        1) Generate a python code to visualize the dataframe.
        2) Consider all necessary libraries and dataframe is imported, just generate only visualization code.
        3)The visualization plot should reveal exact information and relation between columns and rows.
        4)Output should only have code for 5 suitable plots and it should be readymade to deploy in streamlit app without any indentation errors and in below format
        5) Plot graphs between categorical vs numerical column or numerical vs numerical. 
        output code format:
        st.subheader(Subplot name)
        fig(plot number), axis(plot number)=plt.subplots()
        sns.(suitable plot)(data=df,x=(suitable row),y=(suitable column),hue=(suitable column))
        plt.xticks(rotation=20,fontweight='light',fontsize=5)
        axis_no.set_xlabel(suitable x axis name)
        axis_no.set_ylabel(suitable y axis name)
        st.pyplot(fig(plot number))


        dataframe:
        ```{df}```
        """
        executable_visualization_code=generate_visualization_code(prompt) #returns code in a string format.
           
        list_of_codes=executable_visualization_code.splitlines() #splitting each line and executing it to display graphs.

        for code in list_of_codes:
            try :
                exec(code)
            except (SyntaxError, ValueError, IndentationError):
                continue
            
        
    


if __name__ == "__main__":
    main()


