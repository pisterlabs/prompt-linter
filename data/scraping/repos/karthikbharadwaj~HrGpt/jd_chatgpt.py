import openai
import streamlit as st

openai.api_key = "replace with api key"


model_engine = "text-davinci-003"
prompt = "Generate a data scientist Job Description"

st.title("HR Job Description Generator")

def submit_callback():


    completion = openai.Completion.create(
        engine=model_engine,
        prompt=st.session_state["query"],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    result = completion.choices[0].text
    st.info(result)
    st.download_button("Download Job Description",result)




role = st.selectbox('Select a role to generate the JD',("Data Scientist",
                     "Data Engineer",
                     "Solution Architect",
                     "Chief Technology Officer"))

exp = st.selectbox("Minimum Experience",(range(1,25)))
specs = st.text_area(label="Add specifications of the role",value="")

st.session_state["query"] = "Generate a job description for " + role + " with minimum experience " + str(exp) + " having skills in " + specs

if st.button("Generate JD", type='primary'):
        # Use GPT-3 to generate a summary of the article
        response = openai.Completion.create(
        engine=model_engine,
        prompt=st.session_state["query"],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
        # Print the generated summary
        res = response["choices"][0]["text"]
        st.success(res)
        st.download_button('Download JD', res)



























