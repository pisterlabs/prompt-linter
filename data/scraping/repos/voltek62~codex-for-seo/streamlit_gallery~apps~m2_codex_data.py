import streamlit as st
import pandas as pd
from shutil import copyfile
from datetime import datetime
import os
import openai
import requests
import json
from io import StringIO
import subprocess
import sys


openai.api_key = st.secrets["API_TOKEN"]

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
    # uniqueID = _get_session()
    
    #data = {"prompt": prompt, "max_tokens": 200, "temperature": 0, "stop": ["# 4."],"user": uniqueID}
    data = {"prompt": prompt, "max_tokens": 200, "temperature": 0, "stop": ["# 4."]}
    
    data.update(kwargs)

    response = requests.post(
        COMPLETION_ENDPOINT, headers=headers, data=json.dumps(data)
    )
    result = response.json()

    if debug:
        print("COMPLETION_ENDPOINT:")
        print(json.dumps(COMPLETION_ENDPOINT, indent=4))
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
    from subprocess import PIPE, Popen

    command = (
        data
        + "\n\n# The filename is "
        + datapath
        + "\n"
        + text
        + "\nimport pandas as pd\ndf = pd.read_csv('"
        + datapath
        + "')\n"
    )

    # create the python script
    str_now = datetime.now().strftime("%Y%m%d%H%M%S")
    scriptpath = os.path.join("scripts/", str_now + "scriptm2s3.py")

    with open(scriptpath, "w") as f:
        res = completion(command)[0]
        f.write(command + "\n" + res)

    code = res

    process = Popen(
        f"{sys.executable} " + scriptpath,
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
    )
    out, error = process.communicate()

    return out, error, code, command


def format_output(out, error, code, prompt):

    st.header("Result")

    st.markdown(
        f"""
```
{out}
```"""
    )
    st.header("Full code:")
    st.markdown(
        f"""
```
{code}
```"""
    )
    st.header("Errors:")
    st.markdown(
        f"""
```
{error}
```"""
    )

    st.markdown("""---""")
    st.header("Prompt sent to Codex")
    st.text(prompt)


def main():

    text_input_container = st.empty()
    new_password = text_input_container.text_input("")

    if not new_password:
        st.warning("Please type your password!")
        st.stop()
    elif "M2S3DATA" in new_password:
        text_input_container.empty()
    else:
        st.warning("🚨 Nope, this password doesn't work")
        st.stop()

    st.title("Data Analysis")

    with st.form(key="myform"):

        uploaded_file = st.file_uploader(
            "Choose a CSV file with explicit columns name", type="csv"
        )

        str_now = datetime.now().strftime("%Y%m%d%H%M%S")

        if uploaded_file is not None:

            try:
                with open(
                    os.path.join("temp/", str_now + uploaded_file.name), "wb"
                ) as f:
                    f.write(uploaded_file.getbuffer())
            except Exception:
                st.warning(
                    "Sorry file was not temporarily stored for upload.Please re-run the process."
                )

        data_example = """#My data dictionnary has 5 columns:
#Variable|Definition
#Queries|string
#Clicks|int
#Impressions|int
#CTR|percentage
#Position|float"""

        text_example = """# 1. Print 'Hello World'
# 2. Group by queries and sum clicks and impressions
# 3. Print the top 10 query with maximum clicks in a variable named 'top_10_clicks'
# 4. Keep only the columns 'Queries' and 'Clicks'
# 5. Use the function print(top_10_clicks)"""

        data = st.text_area("Describe your data ?", data_example, height=300)

        text = st.text_area("What do you want ?", text_example, height=20)

        submitted = st.form_submit_button(label="Execute")

        if submitted:

            # if uploaded_file is None:

            col1, col2, col3 = st.columns(3)
            with col2:
                gif_runner = st.image("images/mouse.gif")

            # by default
            if uploaded_file is None:
                datapath = "examples/data.csv"
            else:
                datapath = os.path.join("temp/", str_now + uploaded_file.name)

            out, error, code, prompt = execute(data, text, datapath)
            gif_runner.empty()
            format_output(out, error, code, prompt)


if __name__ == "__main__":
    main()
