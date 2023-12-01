#for comptetion in build. Joey,
#from team TDI-India, INDIA.
'''1-Create a program that analyses any random github repo and provides easy and quick explantion of code\n, 
with special reference to mathmatical logic behind code. 
2- Analyse the users metric, analyse how many times he asks ai and then gives a good rating,'''
#temp weblink wwww.getrightgit.com

#use cases target - any coder , primary focus on PYTHON codes

import streamlit as st 
import os

import requests
import json
import openai
# from langchain.embeddings import OpenAIEmbeddings
# from openai.embeddings_utils import cosine_similarity
import os
import pandas
from streamlit_lottie import st_lottie 
import subprocess
import queue
from analyze_code import get_repo
repo_queue = queue.Queue()





# Create a Streamlit app with a red title
st.markdown("<h1 style='color: green;'>Get the right Git-Repo! problem?</h1>", unsafe_allow_html=True)

with open('coder.json','r') as f:
    image = json.load(f)
# Display the Lottie animation
st_lottie(image,height=200, 
        width=200)



#user inputs  
problem_statement_user =[] 


l1 = st.text_input("Tell us your problem", value="Describe clearly your problem", key="problem_input")
if l1:
    problem_statement_user.append(l1)
else:
    pass

l2=st.text_input("what github repo you want to use for solving above issue", value="provide link")



sp_file_input=st.text_input("specific file which you want to analyze", value="provide link")

#user action.
b3=st.button("analyze code")
if b3:
    print("github link", l2)
    if l2 :
        # Run the OpenPose command asynchronously
        command = (
            f"git clone {l2}"
            
        )

        # Start the command without waiting for completion
        repo_queue.put(l2)
        process = subprocess.Popen(command, shell=True)
        with open('downloading.json','r') as f:
            image = json.load(f)
        # Display the Lottie animation
        st_lottie(image,height=200, 
            width=200)
        process.wait()  # Wait for the process to finish
    
    # Check if the process completed successfully
        if process.returncode == 0:
            st.write("Done")  # Print "Done" if the download is successful
            get_repo(repo_queue)
        else:
            st.write("Download failed")  # Print an error message if the download failed
       

    else:
        pass
