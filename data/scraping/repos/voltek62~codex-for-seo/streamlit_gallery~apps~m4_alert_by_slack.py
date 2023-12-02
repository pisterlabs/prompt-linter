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
    #"user": uniqueID,
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


def execute(data, text, datapath):

    command = (
        data
        + "\n\n# The filename is "+datapath+"\n"
        + text
        + "\n\n\nimport pandas as pd\ndf = pd.read_csv('"+datapath+"')\n"
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
    elif "M4S4SLACK" in new_password:
        text_input_container.empty()
    else:
        st.warning("🚨 Nope, this password doesn't work")
        st.stop()

    with st.form(key="myform"):
        
        st.title("Being alerted by Slack")
        
        """
        uploaded_file = st.file_uploader("Choose a CSV file with explicit columns name",type="csv")
        
        str_now = datetime.now().strftime("%Y%m%d%H%M%S")   
        
        if uploaded_file is not None:
    
            try:
                with open(os.path.join("temp/",str_now+uploaded_file.name),"wb") as f:
                    f.write(uploaded_file.getbuffer())
            except  Exception:
                st.warning("Sorry file was not temporarily stored for upload.Please re-run the process.")
        """

        data_example = """#My data dictionnary has 3 columns :
#Variable|Definition
#3xx|3xx errors
#4xx|4xx errors
#5xx|5xx errors"""

        text_example = """# Create a python script to send slack message after reading CSV file where a value in the colunm '4xx errors' is greater than 10.
# 1. The message is ‘Too many 4xx errors are  detected'
# 2. My slack webhook URL is 'https://hooks.slack.com/services/T8X0XXBU8/'"""

        data = st.text_area("Describe your data ?", data_example, height=140)

        text = st.text_area("What do you want ?", text_example, height=60)

        submitted = st.form_submit_button(label="Execute")

        if submitted:
            
            col1, col2, col3 = st.columns(3)
            with col2:
                gif_runner = st.image("images/mouse.gif")
                
            # by default
            datapath = "examples/m4_alert_by_slack.csv"
            
            """
            if uploaded_file is None:
                datapath = "examples/m4_alert_by_slack.csv"
            else:
                datapath = os.path.join("temp/",str_now+uploaded_file.name)
            """
            
            code = execute(data, text, datapath)
            
            gif_runner.empty()
            
            format_output(code)
            
            # CREATE IPYNB and host on server
            # ipython nbconvert --to python mnist.py mnist.ipynb
            
            link='<a target="_blank" href="https://deepnote.com/launch?name=Slack%20Alert&url=https://company.com/link/to/static/notebook.ipynb"><img src="https://deepnote.com/buttons/launch-in-deepnote.svg"></a>'
            st.markdown(link,unsafe_allow_html=True)
            
            

if __name__ == "__main__":
    main()
