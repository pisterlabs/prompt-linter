
import openai
import streamlit as st


openai.api_key =  st.secrets["OPENAI_API_KEY"]

st.title("Isolated Falcons")

st.write("""
    ## Summarizer
"""
)

article_text = st.text_area("Enter the text you want to summarize :", height=130)



if len(article_text)>100:
    
    temp = st.slider("Set the creativity of the summary :", min_value=0.0, max_value=1.0, step=0.1)
    token = st.slider("Set the length of the summary :", min_value=50, max_value=1000, step=50)
    if st.button("Generate Summary"):
        respose = openai.Completion.create(
            engine = "text-davinci-003",
            prompt = "Please summarize the following article in few sentences : " +  article_text ,
            max_tokens = token,
            temperature = temp,
        )

        res = respose.choices[0].text
        st.info(res)

        st.download_button(
            label="Download Summary",
            data=res,
            file_name="summary.txt",
            mime="text/plain",
            )
else:
    st.warning("Please enter a text longer than 100 characters")
