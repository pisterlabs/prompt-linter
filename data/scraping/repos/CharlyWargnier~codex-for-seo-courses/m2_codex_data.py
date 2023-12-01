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


def completion(prompt, engine_id="davinci-codex", debug=False, **kwargs):

    COMPLETION_ENDPOINT = (
        "https://api.openai.com/v1/engines/{engine_id}/completions".format(
            engine_id=engine_id
        )
    )

    headers = {
        "Authorization": "Bearer {api_key}".format(api_key=openai.api_key),
        "Content-Type": "application/json",
    }

    data = {"prompt": prompt, "max_tokens": 200, "temperature": 0, "stop": ["# 4."]}

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
        # "python "+scriptpath,
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

        data_example = """#My data dictionnary has 3 columns :
#Variable|Definition|Key
#survival|Survival|0 = No, 1 = Yes
#pclass|Ticket class|1 = 1st, 2 = 2nd, 3 = 3rd
#sex|Sex|	
#Age|Age in years|	
#sibsp|# of siblings / spouses aboard the Titanic|	
#parch|# of parents / children aboard the Titanic|	
#ticket|Ticket number|	
#fare|Passenger fare|	
#cabin|Cabin number|	
#embarked|Port of Embarkation|C = Cherbourg, Q = Queenstown, S = Southampton"""

        text_example = """# Use the print function for displaying
# 1. Print the number of passengers who survived, grouped by Pclass
# 2. Display the median Age of passengers
# 3. Display the Age of passengers, grouped by Sex   
    """

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
