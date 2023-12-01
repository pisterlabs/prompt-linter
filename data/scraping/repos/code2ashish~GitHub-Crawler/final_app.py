import git
import os 
import requests
import json
import openai 
# from dotenv import find_dotenv,load_dotenv
from collections import Counter
import re
import time 

import streamlit as st

st.title('Github Crawler🤖')


def test_openai_key():
    try:
        response = openai.Completion.create(
            engine='davinci',
            prompt='Hello, OpenAI!',
            max_tokens=5
        )

        if 'choices' in response:
            return True
        else:
            return False
    except :
        return False
# load_dotenv(find_dotenv)

c1,c2 = st.columns(2)

with c1:

    key = st.text_input('Enter OpenAI API KEY [We will not store your key] as "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"', '',)
    openai.api_key = key
    open_key =False
    if key!='':
        open_key=True


with c2:
    link_aaddress = st.text_input('Enter link Github User as 🔗"https://github.com/USER_NAME" or "USER_NAME"', '').split('/')
    sub =False
    sub = st.button('Submit')
# username = "code2ashish"
if open_key and test_openai_key():



    if len(link_aaddress)>=1:
        username = link_aaddress[-1]
        print(username)
    if username != '' and sub :
        st.header('Welcome 🎊'+username)
        t=st.empty()
        my_bar = st.progress(0)

    # for percent_complete in range(100):
    #     t.write(percent_complete)
    #     time.sleep(0.2)
    #     my_bar.progress(percent_complete + 1)
    #     st.empty()


        repos = []

        # # Create a URL to the GitHub API endpoint
        url = "https://api.github.com/users/{}/repos".format(username)

        # Make a GET request to the API endpoint
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:

            # Get the response data as JSON
            data = json.loads(response.content)

            # Iterate over the repositories
            for repo in data:

                # Print the repository name
                # print(repo["name"])
                repos.append(repo["name"])
                

        else:

            st.error("Error: HTTP {}.".format(response.status_code))





        def read_file(file_path):
            with open(file_path, 'r') as file:
                contents = file.read()
            return contents

        def extract_names_from_code(file_path):
            if file_path.endswith(".py"):
                with open(file_path, 'r') as file:
                    code = file.read()

                # Extract library names
                library_names = re.findall(r'import\s+(\w+)', code)

                # Extract function names
                function_names = re.findall(r'def\s+(\w+)', code)

                # Extract class names
                class_names = re.findall(r'class\s+(\w+)', code)

                return library_names, function_names, class_names
            if file_path.endswith(".ipynb"):
                # Get the libraries
                libraries = re.findall(r"import\s+([\w]+)", open(file_path).read())

                # Get the function names
                function_names = re.findall(r"def\s+([\w]+)\s*\((.*?)\)", open(file_path).read())

                # Get the class names
                class_names = re.findall(r"class\s+([\w]+)\s*:\s*", open(file_path).read())

                # Return the results
                return libraries, function_names, class_names


        def fetch_files_and_folders(folder_path,json={}):

            folders = []
            library_names=[]
            function_names =[]
            class_names = []

            
            # Iterate through the contents of the folder
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                
                if os.path.isfile(item_path):  # Check if it's a file
                    try:
                        file_contents = read_file(item_path)
                        l,f,c = extract_names_from_code(item_path)

                        library_names.extend(l)
                        function_names.extend(f)
                        class_names.extend(c)
                    except:
                        pass
                elif os.path.isdir(item_path):  # Check if it's a folder
                    folders.append(item)

            # Iterate through the folders and fetch files recursively

            for folder in folders:

                for item in os.listdir(os.path.join(folder_path, folder)):
                    item_path = os.path.join(folder_path, folder, item)

                    if os.path.isfile(item_path):  # Check if it's a file
                        try:
                            file_contents = read_file(item_path)
                            l,f,c = extract_names_from_code(item_path)
                            library_names.extend(l)
                            function_names.extend(f)
                            class_names.extend(c)
                        except:
                            pass
                
                    elif os.path.isdir(item_path):  # Check if it's a folder
                        pass


            return library_names, function_names, class_names

        




        def chat_with_chatbot(prompt):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a user seeking advice on programming language."},
                    {"role": "user", "content": prompt}
                ],
                temperature = 0
            )

            chatbot_response = response.choices[0].message.content
            return chatbot_response





        final_complexity ={}





        time_stamp = time.time()
        count =1
        
        for r in repos:
            t.write(f'[{count}/{len(repos)}]'+'     Analysing :'+r)
            
            
            
        # clonning of the repos
            repo_url = f"https://github.com/{username}/{r}.git"
            destination_folder = f"store/{username}/"
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
                
            
            destination_folder1 = os.path.join(destination_folder,r)
            if not os.path.exists(destination_folder1):
                os.makedirs(destination_folder1)
                git.Repo.clone_from(repo_url, destination_folder1)

            library_names, function_names, class_names = fetch_files_and_folders(destination_folder1,json={})

            prompt = f"""comment about the complexity of the code that have these libraries and functions  and classes
            Libraries:
            {str(Counter(library_names))[8:-1]}
            Functions:
            {str(Counter(function_names))[8:-1]}
            Classes:
            {str(Counter(class_names))[8:-1]}

            strictly choose among 'very simple', 'simple', 'moderate', 'high', 'very high'
            only no explanation needed and only and only one among these """

            user_input = prompt
            if time.time() - time_stamp <30:
                time.sleep(20)
                
            response = chat_with_chatbot(user_input)
            time_stamp = time.time()
            
            
            st.write(r+':'+response)
            final_complexity[r] = response
            
            my_bar.progress(round(count/len(repos)*100))
            count += 1
            t.empty()
            
            
            
            






        prompt = f'''json that has key the repository name and complexity corresponding  to that in values
        you need to find the highest complexity repo

        {final_complexity}

        no need of code and explanation just repo name only'''



        response = chat_with_chatbot(prompt)
        st.success(response)
        st.balloons()

else:
    st.error('Please Provide Valid Open API Key')


