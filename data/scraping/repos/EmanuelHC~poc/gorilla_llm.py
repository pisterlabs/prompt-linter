import openai
import streamlit as st
import subprocess

#openai.api_key = "EMPTY" # Key is ignored and does not matter
#openai.api_base = "http://zanino.millennium.berkeley.edu:8000/v1"


#Query Gorilla Server
def get_gorilla_response(prompt, model):
    try:
        completion = openai.ChatCompletion.create(
            model = model,
            messages = [{"role": "user", "content": prompt}]
        )
        print("Response: ", completion)
        return completion.choices[0].message.content 
    except Exception as e:
        print("Sorry, something went wrong!")

def extract_code_from_output(output):
    code =output.split("code>>>:")[1]
    return code

def run_generated_code(file_path):

    # Command to run the generated code using Python interpreter
    command = ["python", file_path]

    try:
        # Execute the command as a subprocess and capture the output and error streams
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the subprocess ran successfully
        if result.returncode == 0:
            st.success("Generated code executed successfully.")
            # Display the output of the generated code
            st.code(result.stdout, language="python")
        else:
            st.error("Generated code execution failed with the following error:")
            # Display the error message
            st.code(result.stderr, language="bash")
    
    except Exception as e:
        st.error("Error occurred while running the generated code:", e)