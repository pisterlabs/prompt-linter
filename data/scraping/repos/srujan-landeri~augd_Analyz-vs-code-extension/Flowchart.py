import openai
import streamlit as st
import streamlit.components.v1 as components

openai.api_key = "Put your API key here"
from gtts import gTTS  # new import
from io import BytesIO
import re

st.set_page_config(layout="wide")


def mermaid_ele(code: str) -> None:
    components.html(
        f"""
        
        <style>
        .mermaid {{
            background-color: white;
            border-radius: 15px;
            padding: 10px;
            display: flex;
            flex-direction: row;
            justify-content: center;
        }}
        </style>
        
        <pre class="mermaid">
            {code}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=1200,
    )


def text_to_speech(text):
    """
    Converts text to an audio file using gTTS and returns the audio file as binary data
    """
    audio_bytes = BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.read()


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inder');
    .container {
        display: flex;
    }
    .logo-text {
        font-family: 'Inder', sans-serif !important;
        font-size:50px !important;
        padding-left:15px !important;
        margin-top: 40px !important;
    }
    
    .logo-img {
        float:left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="https://cdn-icons-png.flaticon.com/128/2593/2593635.png">
        <div class="logo-text">Visualyz with Analyz!!!</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Get prompt from user
prompt = st.text_input(
    "Ask away and get an easy to read representation of what you need!:"
)

# On prompt submission, send request to OpenAI API
if st.button("Ask Analyz"):
    with st.spinner("Analyzing..."):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"""
            You are a teacher and your student has asked you a question: {prompt}. 
            Please provide a detailed theoretical answer in bullet points. 
            You must always give an answer.
            Also, create a Mermaid.js diagram to visually represent the information. 
            The diagram should be formatted as code and contain at least 15 nodes. 
            It should be top to bottom orientation ie, 'graph TD;' should be the starting line of the graph.
            Feel free to add as many nodes as necessary and cycles if needed. Make use of labels and tooltips to make the diagram more readable.
                     
             
            After viewing the diagram, the student should have no further questions.
            Please start the Mermaid.js code with ‘MERMAID_START’ and end it with ‘MERMAID_END’. 
            The diagram should be the last part of the answer, not inserted in the middle."
            """,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.7,
        )

        col1, col2 = st.columns(2)
        text_body = ""

        with col1:
            # Extract mermaid code from response
            mermaid_code = response.choices[0]["text"]
            text_data = mermaid_code.split("MERMAID_START")
            # print(text_data)
            corpus = text_data[0].split("\n")
            for i in corpus:
                text_body += i
                st.write(i.strip())
            st.audio(text_to_speech(text_body), format="audio/wav")

        with col2:
            start_marker = "MERMAID_START"
            end_marker = "MERMAID_END"
            start_index = response.choices[0].text.find(start_marker)
            end_index = response.choices[0].text.find(end_marker)
            if start_index != -1 and end_index != -1:
                mermaid = (
                    response.choices[0]
                    .text[start_index + len(start_marker) : end_index]
                    .strip()
                )
            else:
                mermaid = "No mermaid.js graph found in the response."

            mermaid_ele(mermaid)
