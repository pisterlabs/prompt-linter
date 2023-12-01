import streamlit as st
import pandas as pd
from shutil import copyfile
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
    "max_tokens": 600, 
    "temperature": 0
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


def execute(text):

    command = "# Create a Python script\n" + text + "\n\n"

    code = completion(command)[0]

    return command + "\n\n" + code


def format_output(code):

    st.title("Full code:")
    st.code(code)


def main():

    text_input_container = st.empty()
    new_password = text_input_container.text_input("")

    if not new_password:
        st.warning("Please type your password!")
        st.stop()
    elif "M4S3EMAIL" in new_password:
        text_input_container.empty()
    else:
        st.warning("🚨 Nope, this password doesn't work")
        st.stop()

    with st.form(key="myform"):

        st.title("Get Stats by email")

        data_example = """# 1. Import data from the daily CSV : https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv 
# The first column is 'Date'. The second column is 'Temp'. The date's format is : YYYY-MM-DD
# 2. Get the last temperature in the dataframe in a variable 'last_temp'
# 3. Send an email with the sendinblue API it the variable 'last_temp' is superior than 10. 
# 4. The API key is 'xkeysib-XXXXXXXXXX-XXXXXXXXXX'
# 5. The title of the email : 'New high temperature detected'
# 6. The receiver is 'receiver@datamarketinglabs.com'
# 7. The campaign name is 'Temp Alert'
# 8. The HTML template is '<html></html>'
# 9. The sender is 'sender@zescore.com'"""

        text = st.text_area("What do you want ?", data_example, height=200)

        submitted = st.form_submit_button(label="Execute")

        if submitted:

            col1, col2, col3 = st.columns(3)
            with col2:
                gif_runner = st.image("images/mouse.gif")

            code = execute(text)

            gif_runner.empty()

            format_output(code)

            link = '<a target="_blank" href="https://deepnote.com/launch?name=Email%20Alert&url=https://company.com/link/to/static/notebook.ipynb"><img src="https://deepnote.com/buttons/launch-in-deepnote.svg"></a>'
            st.markdown(link, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
