import streamlit as st
import pandas as pd
from shutil import copyfile
from datetime import datetime
import os
import openai
import requests
import json
from io import StringIO

# openai.organization = ""
openai.api_key = st.secrets["API_TOKEN"]
# def login_form():
#     list_of_strings = []
#
#     password = st.sidebar.text_input("Password", type="password")  # on_change=set_pw
#
#     type(password)
#
#     some_list = ["abc_123", "def_456", "ghi_789", "abc_456"]
#
#     if any(password in s for s in some_list):
#         st.success("✅ Yes, this password works!")
#         app = st.button("💫 I'm an app!")  # on_change=set_pw
#
#         st.stop()
#
#     else:
#         st.warning("🚨 Nope, this password doesn't work")
#         st.stop()


def build_prompt(words):

    prompt = """
  {words}
  """

    prompt = prompt.format(words=words)

    return prompt

from streamlit.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.server.server import Server

def _get_session():
    session_id = get_script_run_ctx().session_id
    session = Server.get_current().get_session_by_id(session_id)
    if session is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return str(session)[50:64]


def completion(prompt, engine_id="code-davinci-002", debug=False, **kwargs):

    COMPLETION_ENDPOINT = (
        "https://api.openai.com/v1/engines/{engine_id}/completions".format(
            engine_id=engine_id
        )
    )

    headers = {
        "Authorization": "Bearer {api_key}".format(api_key=openai.api_key),
        "Content-Type": "application/json",
    }
    
    # Pass a uniqueID for every user w/ each API call (both for Completion & the Content Filter) e.g. user= $uniqueID. 
    # This 'user' param can be passed in the request body along with other params such as prompt, max_tokens etc.
    #uniqueID = _get_session()    

    data = {"prompt": prompt, 
    "max_tokens": 200, 
    "temperature": 0, 
    "stop": ["# 4."]
    #"user": uniqueID
    }

    data.update(kwargs)

    response = requests.post(
        COMPLETION_ENDPOINT, headers=headers, data=json.dumps(data)
    )
    result = response.json()

    if debug:
        print("Headers:")
        print(json.dumps(headers, indent=4))
        print("Data:")
        print(json.dumps(data, indent=4))
        print("Result:")
        print(json.dumps(result, indent=4))

    if response.status_code == 200:
        return [x["text"].strip() for x in result["choices"]]
    else:
        return "Error: {}".format(result["error"]["message"])


def execute(data, text):

    command = (
        data
        + "\n\n# The filename is data.csv\n"
        + text
        + "\n\n\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n"
    )

    code = completion(command)[0]

    return command+'\n\n'+code


def format_output(code):
    st.title("Full code:")
    st.code(code)


def main():

    text_input_container = st.empty()
    new_password = text_input_container.text_input("")

    if not new_password:
        st.warning("Please type your password!")
        st.stop()
    elif "M4S2WP" in new_password:
        text_input_container.empty()
    else:
        st.warning("🚨 Nope, this password doesn't work")
        st.stop()

    with st.form(key="myform"):
        
        st.title("Post Generated text in a WP")
        
        data_example = """#My data dictionnary has 2 columns :
#Variable|Definition
#title|Title of my blog post
#content|Content"""
        
        wp_endpoint = st.text_input("WORDPRESS_BLOG_API_ENDPOINT (without /posts)","https://monblog.fr/v2")
        wp_user = st.text_input("WORDPRESS_USER","user")
        wp_password = st.text_input("WORDPRESS_APP_PASSWORD","xxxx")
        
        # TODO : force the usage of my function

        text_example = """# Create a python script to read each line of CSV and send the content to my website with the WordPress API.\n
# Use the following function to send the content : post_to_wordpress_blog(title, content)
def post_to_wordpress_blog(title, content):

    data_string = 'WORDPRESS_USER:WORDPRESS_APP_PASSWORD'
    token = base64.b64encode(data_string.encode())
    headers = {'Authorization': 'Basic ' + token.decode('utf-8')}
    
    days_ago = random.randint(1, 7)
    pub_datetime = datetime.now() - timedelta(days=days_ago)
    iso_date = pub_datetime.isoformat()

    post = {
        'date': iso_date,
        'title': title,
        'status': 'publish',
        'content': content,
        'author': 1,
        'format': 'standard',
        'tags': []
    }

    # pass Custom Field
    endpoint = 'WORDPRESS_BLOG_API_ENDPOINT/posts'

    r = requests.post(endpoint, headers=headers, json=post, timeout=60, allow_redirects=False)

    print(f"Posted to Wordpress.")
        """
        
        text_example = text_example.replace("WORDPRESS_BLOG_API_ENDPOINT",wp_endpoint)
        text_example = text_example.replace("WORDPRESS_USER",wp_user)
        text_example = text_example.replace("WORDPRESS_APP_PASSWORD",wp_password)        
        
        data = st.text_area("Describe your data ?", data_example, height=140)

        text = st.text_area("What do you want ?", text_example, height=60)

        submitted = st.form_submit_button(label="Execute")

        if submitted:
            
            col1, col2, col3 = st.columns(3)
            with col2:
                gif_runner = st.image("images/mouse.gif")  
                
            code = execute(data, text)
            
            gif_runner.empty()
            
            format_output(code)
            
            link='<a target="_blank" href="https://deepnote.com/launch?name=WP%20Import&url=https://company.com/link/to/static/notebook.ipynb"><img src="https://deepnote.com/buttons/launch-in-deepnote.svg"></a>'
            st.markdown(link,unsafe_allow_html=True)            


if __name__ == "__main__":
    main()
