## importing all the modules
import os
import requests
import subprocess
import nbformat
import tempfile
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.document_loaders import TextLoader
import streamlit as st
from fileprocessing import repoaccess, extract_code_cells, format_code_cells, is_float, delete_files_except_extensions
import dotenv


dotenv.load_dotenv(dotenv_path="key.env")
#Api key
#streamlit framework for frontend
st.title("Github Analysis Project")
profile=st.text_input("Enter the Github Profile link")

if profile:
    repos=repoaccess(profile) # list of all repositories in the github profile
    local_path="temp_clone"    # a temporary path for cloning temporarily.
    print(repos)
    ## AI prompt template
    template = """/
    Calculate and return the summation of cyclomatic complexity*1,nesting depth*0.25, lines of code*0.005 o{code}
    Instr:
    1.Return the total value of summationonly in the format strictly without any working, just return the final value of summation even if it reaches to more than one decimal places,dont return any working just the float value of summation.
    Answer: 
    """
    prompt = PromptTemplate.from_template(template)
    complex={}
    print(repos)
    for repo in repos:
        subprocess.check_output(['rm', '-rf', local_path])
        try:
            print(repo)
            subprocess.run(['git', 'clone', repo, local_path], check=True)
            print("true")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            print("false")
        extensions = ['.py', '.js', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php', '.scala','.sh', '.bash', '.sql', '.ipynb']
        delete_files_except_extensions(local_path,extensions)
        docs=[]
        for filename in os.scandir(local_path): 
            if filename.is_file():
                print((filename.name))
                try:
                    loader = None
                    if filename.name.endswith(".ipynb"):
                        code_cells = extract_code_cells(filename)
                        formatted_code = format_code_cells(code_cells)
                        docs.append(formatted_code)
                    else:
                        loader = TextLoader(filename)
                        x=loader.load()
                        docs.append(x[0].page_content)
                except Exception as e:
                    print("Error loading files with pattern")
                    continue 
        print(len(docs))

        complexity=0.0
        for f in docs:
            print(len(f),f)
            newtemp=prompt.format(code=f)
            llm=OpenAI(temperature=0.5)
            result=llm(newtemp)
            if(is_float(result)):
                val=(float)(result)
                complexity+=val
        print(repo,complexity)
        complex[repo]=complexity
    sortedDict = dict(sorted(complex.items(), key=lambda x:x[1]))
    subprocess.check_output(['rm', '-rf', local_path])
    st.write(list(sortedDict)[-1])


