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
from langchain.embeddings import OpenAIEmbeddings

import os
import pandas
from streamlit_lottie import st_lottie 
import subprocess
import queue
from analyze_code import get_repo, analysis_by_ai
repo_queue = queue.Queue()





# Create a Streamlit app with a red title
st.markdown("<h1 style='color: green;'>Get the right Git-Repo! problem?</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='color: red;'>Currently only Supports Python codes</h1>", unsafe_allow_html=True)

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

# make a folder to upload user github repo

if not os.path.exists("user_github"):
    os.mkdir("user_github")
else:
    pass

#user action.
b3=st.button("analyze code")
if b3:
    print("github link", l2)
    if l2 :
         # Create a placeholder for the Lottie animation
        downloading_placeholder = st.empty()
        # Run the OpenPose command asynchronously
        command = (
            f"cd user_github && git clone {l2}"
            
        )

        # Start the command without waiting for completion
        repo_queue.put(l2)

        process = subprocess.Popen(command, shell=True)
        with open('downloading.json','r') as f:
            image = json.load(f)
        # Display the Lottie animation
        downloading_placeholder.write(st_lottie(image, height=200, width=200))
        
        process.wait()  # Wait for the process to finish


        # Try to empty the placeholder and catch any exception that occurs
        try:
            downloading_placeholder.empty()
        except Exception as e:
            st.error(f"An error occurred while trying to clear the animation: {e}")
       
    
    # Check if the process completed successfully
        if process.returncode == 0:
            st.write("Done")  # Print "Done" if the download is successful
            
            with open("correct.json",'r') as f:
                correct = json.load(f)
              # Display the Lottie animation
            downloading_placeholder.write(st_lottie(correct, height=100, width=100))
            list_of_files = get_repo(repo_queue)
            st.write("Total files found in repo",len(list_of_files))
            st.write("Exact Files with names", list_of_files)
            content_python_files =analysis_by_ai(problem_statement_user, list_of_files,repo_queue)
            print("content_python_files", content_python_files)

        else:
            st.write("Download failed")  # Print an error message if the download failed
       

    else:
        pass
    
    





