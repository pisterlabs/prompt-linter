import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv, set_key
import openai
from bs4 import BeautifulSoup
from datetime import datetime
import textwrap
import csv
import random
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv, set_key
import pandas as pd
import os
import csv
import openai
from bs4 import BeautifulSoup
from datetime import datetime
from io import BytesIO, StringIO

st.set_page_config(layout="wide", page_icon="🇺🇸")
st.image("static/assets/SpeedCandidating.png", use_column_width=True)
readme_placeholder = st.empty()
if 'readme_displayed' not in st.session_state:
    st.session_state['readme_displayed'] = True
if 'research_button_clicked' not in st.session_state:
    st.session_state['research_button_clicked'] = False
if 'chat_button_clicked' not in st.session_state:
    st.session_state['chat_button_clicked'] = False

if st.session_state['readme_displayed']:
    readme_placeholder = st.empty()  

    readme_content = """

![GitHub last commit](https://img.shields.io/github/last-commit/NoDataFound/SpeedCandidating)
![GitHub issues](https://img.shields.io/github/issues-raw/NoDataFound/SpeedCandidating)
![GitHub pull requests](https://img.shields.io/github/issues-pr/NoDataFound/SpeedCandidating)
![GitHub stars](https://img.shields.io/github/stars/NoDataFound/SpeedCandidating?style=social)
![OpenAI API](https://img.shields.io/badge/OpenAI%20API-B1A6F0.svg?style=flat-square&logo=openai)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat-square&logo=streamlit)
![Pandas](https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas)
![Plotly](https://img.shields.io/badge/plotly-3F4F75.svg?style=flat-square&logo=plotly)
![Python](https://img.shields.io/badge/python-3776AB.svg?style=flat-square&logo=python&logoColor=ffffff)

SpeedCandidating is an interactive application designed to engage users in political discourse, allowing them to ask questions and receive responses from virtual representations of political candidates trained from official sources. Powered by OpenAI's GPT models, it aims to eliminate media bias and simulate a candidating session where users can quickly gather insights into various political personas.

## Features

- **Multiple Party Interaction**: Engage with candidates across different political parties.
- **Dynamic Questioning**: Ask questions and get personalized responses from the candidate's perspective.
- **Data Logging**: Keeps track of all questions and responses for further analysis.

Visit  https://github.com/NoDataFound/SpeedCandidating  to learn more.
"""

    readme_placeholder.markdown(readme_content)

#load_dotenv('.env')
openai.api_key = st.secrets["OPENAI"]["OPENAI_API_KEY"]

#openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    st.error("OpenAI API key is missing. Please add it to your secrets.")
#if not openai.api_key:
#    openai.api_key = st.text_input("Enter OPENAI_API_KEY API key")
#    set_key('.env', 'OPENAI_API_KEY', openai.api_key)

os.environ['OPENAI_API_KEY'] = openai.api_key

with open("static/assets/css/ssc.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

PARTY_COLORS = {
    'Democrats': '#0a5f8a',
    'Republicans': '#c93e34',
    'Independent': 'white'
}

CANDIDATES = {
    'Democrats': ['Biden', 'Williamson', 'Uygur'],
    'Republicans': ['Trump', 'Haley', 'Ramaswamy', 'Hutchinson', 'Elder', 'Binkley', 'Scott', 'DeSantis', 'Pence', 'Christie', 'Burgum'],
    'Independent': ['Kennedy', 'West']
}

DATA_FILE = "log/questions_responses_log.csv"

def get_party(candidate):
    for party, candidates in CANDIDATES.items():
        if candidate in candidates:
            return party
        
    return None

def log_question(candidates, party, question, response):
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["candidate", "party", "question", "response"])
    else:
        df = pd.DataFrame(columns=["candidate", "party", "question", "response"])

    for candidate in candidates:
        new_data = pd.DataFrame({
            "candidate": [candidate],
            "party": [party],
            "question": [question],
            "response": [response]
        })
        df = df.append(new_data, ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def get_candidate_text(candidate): 
    formatted_name = candidate.replace(' ', '_')
    file_path = f'training/candidates/{formatted_name}.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        
        return file.read()

def get_response(candidate, question, text, is_new_session=False):
    MAX_CHUNK_SIZE = 16000  # Example value, adjust as needed

    selected_persona = (
    f"Ignore all the instructions you got before. From now on, you are going to act as  {candidate}. "
    f"You are talking to a voter"
    f"Respond to questions in the first person as if you are {candidate}, "
    f"using the voice and demeanor of a political figure. Do not refer to yourself in the 3rd person"
    f"Do not ever mention wikipedia"
    f"Try to use bullet points if possible")



    if len(text.split()) <= MAX_CHUNK_SIZE:
        question_with_text = text + " " + question
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {
                    "role": "system",
                    "content": f"{selected_persona}"
                },
                {
                    "role": "user",
                    "content": question_with_text  # Use the prefixed question here
                }
            ],
            temperature=1,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer = response['choices'][0]['message']['content'].strip()
        text_snippets = text.split('. ')
        relevant_snippets = random.sample(text_snippets, min(3, len(text_snippets)))  # Get up to 3 snippets
        
        # Construct the reference section
        reference_section = "\n\nReference: "
        reference_section += f"\n- My answer was derived from: training/candidates/{candidate.replace(' ', '_')}.txt"
        for snippet in relevant_snippets:
            reference_section += f"\n- {snippet}"

        return answer + reference_section
    
    else:
        text_chunks = textwrap.wrap(text, width=MAX_CHUNK_SIZE, expand_tabs=False, replace_whitespace=False, drop_whitespace=False)
        combined_answers = ""
        for chunk in text_chunks:
            question_with_chunk = chunk + " " + question  # Prefix the question with the chunk
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=[
                    {
                        "role": "system",
                        "content": f"{selected_persona}"
                    },
                    {
                        "role": "user",
                        "content": question_with_chunk  # Use the prefixed question here
                    }
                ],
                temperature=1,
                max_tokens=200,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            answer = response['choices'][0]['message']['content'].strip()
            # Randomly select a snippet from the chunk as a reference
            snippet = random.choice(chunk.split('. '))
            reference_section = f"\n\nReference: This answer was derived from: {snippet}"
            combined_answers += answer + reference_section + " "

        return combined_answers


def get_response_table(responses):
    df = pd.DataFrame(responses.items(), columns=["Candidate", "Response"])
    df["Party"] = df["Candidate"].apply(get_party)
    # Rearrange columns
    df = df[["Party", "Candidate", "Response"]]
    return df

def display_table(df):
    # Replace newline characters with HTML line break tag
    df['Response'] = df['Response'].str.replace('\n', '<br>')

    # Convert DataFrame to HTML
    html = df.to_html(classes='table table-sm', escape=False, index=False, border=0, justify='left', header=True)

    # Use BeautifulSoup to manipulate the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Update header row with 'active' class for a lighter color and uppercase, bold text
    header_row = soup.find('tr')
    header_row['class'] = 'active'
    for th in header_row.find_all('th'):
        th.string = th.text.upper()
        th['style'] = 'font-weight: bold;'

    # Update each data row with the appropriate class based on the party
    for tr, party in zip(soup.find_all('tr')[1:], df['Party']):  # Skip header row
        tr['class'] = 'table-danger' if party == 'Republicans' else 'table-info' if party == 'Democrats' else ''

    # Convert back to HTML and then to markdown
    html = str(soup)
    st.markdown(html, unsafe_allow_html=True)

def main():
    if 'research_button_clicked' not in st.session_state:
        st.session_state['research_button_clicked'] = False
    if 'chat_button_clicked' not in st.session_state:
        st.session_state['chat_button_clicked'] = False
    
    col1, col2, col3 , col4, col5, col6 = st.columns([1, 1, 2,2,1,1], gap="medium")
    
    if col3.button("Research Multiple Candidates", key="research_button"):
        st.session_state['research_button_clicked'] = True
        st.session_state['chat_button_clicked'] = False
        st.session_state['readme_displayed'] = False  

    if col4.button("Chat with Individual Candidates", key="chat_button"):
        st.session_state['chat_button_clicked'] = True
        st.session_state['research_button_clicked'] = False
        st.session_state['readme_displayed'] = False  

    if not st.session_state['readme_displayed']:
        readme_placeholder.empty()  
    
    st.markdown("----------")

    new_chat_button_style = """
        <link
            rel="stylesheet"
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
        >
        <style>
            .big-button {
                font-size: 20px;
                padding: 20px 40px;
                margin: 5px 0;
            }
            .big-button:hover {
                color: black !important;
            }
        
        <style>
        .stButton > button {
               background-color: #008CBA; /* Blue color */
            border: none;
            color: black !important;
            hover: black;
            padding: 8px 16px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            }
        </style>
        """

    st.markdown(new_chat_button_style, unsafe_allow_html=True)

    if st.session_state['research_button_clicked']:
        with st.sidebar:
            st.image(os.path.join("static", "assets", "SpeedCandidating.png"), use_column_width=True)

            selected_party = st.selectbox('Select party:', list(CANDIDATES.keys()))
            selected_candidates = st.multiselect(f'Choose {selected_party} candidates:', CANDIDATES[selected_party])
            if selected_party  == 'Democrats':
                st.markdown("""<style>span[data-baseweb="tag"] {  background-color: #242529 !important;}</style>""",unsafe_allow_html=True,)
            if selected_party == 'Republicans':
                st.markdown("""<style>span[data-baseweb="tag"] {  background-color: #242529 !important;}</style>""",unsafe_allow_html=True,)

            additional_party_option = st.checkbox("Select another party?")

            if additional_party_option:
                remaining_parties = [party for party in CANDIDATES.keys() if party != selected_party]
                additional_party = st.selectbox('Select another party:', remaining_parties)
                additional_candidates = st.multiselect(f'Choose {additional_party} candidates:', CANDIDATES[additional_party])
                selected_candidates.extend(additional_candidates)

        with st.form("Ask Question"):
            question = st.text_input(label='',placeholder ="Ask your question")
            if selected_candidates:
                cols = st.columns(len(selected_candidates))
                for idx, candidate in enumerate(selected_candidates):
                    party_of_candidate = get_party(candidate)
                    img_path = os.path.join("resources", "images",f"{party_of_candidate}", f"{candidate.lower()}.png")
                    cols[idx].image(img_path, caption=candidate, width=60)
            ask_all = st.checkbox("Ask all Presidential candidates")
            submit = st.form_submit_button("Submit")

            if submit and question:
                responses = {}
                for candidate in selected_candidates:
                    candidate_text = get_candidate_text(candidate)
                    response = get_response(candidate, question, candidate_text)
                    responses[candidate] = response
                    log_question([candidate], get_party(candidate), question, response)

                # Get the DataFrame and display it
                response_df = get_response_table(responses)
                display_table(response_df)

        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            col1, col2 = st.columns(2)

            candidate_counts = df['candidate'].value_counts()
            candidate_colors = [PARTY_COLORS[get_party(candidate)] for candidate in candidate_counts.index]

            fig1 = go.Figure(data=[go.Bar(x=candidate_counts.index, y=candidate_counts, marker_color=candidate_colors)])
            fig1.update_layout(title="Question Counts per Canidate")
            col1.plotly_chart(fig1, use_container_width=True)

            party_counts = df['party'].value_counts()
            fig2 = go.Figure(data=[go.Pie(labels=party_counts.index, values=party_counts, hole=.3, marker_colors=[PARTY_COLORS[p] for p in party_counts.index])])
            fig2.update_layout(title="Party Question Distribution")
            col2.plotly_chart(fig2, use_container_width=True)

    elif st.session_state['chat_button_clicked']:
        
        st.sidebar.image(os.path.join("static", "assets", "SpeedCandidating.png"), use_column_width=True)
        selected_candidate = st.selectbox('Select a candidate:', ["Candidate"] + [candidate for party in CANDIDATES.values() for candidate in party])
        party_of_candidate = get_party(selected_candidate)
        img_path = os.path.join("resources", "images", f"{party_of_candidate}", f"{selected_candidate.lower()}.png")
        
        import base64

        def image_to_base64(img_path):
            with open(img_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        st.sidebar.markdown("----")
        col1, col2 = st.columns(2)

        with col1:
            st.sidebar.markdown(
                f"<div style='text-align: center; margin: auto;'>CURRENTLY CHATTING WITH</div>", 
                unsafe_allow_html=True
            )

        with col2:
            st.sidebar.markdown(
                f"<div style='text-align: center;'><img src='data:image/png;base64,{image_to_base64(img_path)}' style='margin: auto;'/></div>",
                unsafe_allow_html=True
            )

        st.sidebar.markdown("----")
        st.sidebar.success(f"All responses derived from: training/candidates/{selected_candidate.replace(' ', '_')}.json")

        #if "session_key" not in st.session_state:
        #    st.session_state.session_key = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #    st.session_state.messages = []
        #    candidate_text = get_candidate_text(selected_candidate)
        #    greeting_response = get_response(selected_candidate, "", candidate_text, is_new_session=True)
        #    st.session_state.messages.append({"role": "assistant", "content": greeting_response})
        if "session_key" not in st.session_state:
            st.session_state.session_key = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages = []

 

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Type your message:")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user",avatar=os.path.join("resources", "images", "None","candidate.png")):
                st.markdown(prompt)

            candidate_text = get_candidate_text(selected_candidate)
            response = get_response(selected_candidate, prompt, candidate_text)
            with st.chat_message("assistant",avatar=os.path.join("resources", "images", f"{party_of_candidate}", f"{selected_candidate.lower()}.png")):
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


        #prompt = st.chat_input("Type your message:")
#
        #if prompt:
        #    st.session_state.messages.append({"role": "user", "content": prompt})
        #    with st.chat_message("user"):
        #        st.markdown(prompt)
        #    candidate_text = get_candidate_text(selected_candidate)
        #    response = get_response(selected_candidate, prompt, candidate_text)
        #    with st.chat_message("assistant",avatar=os.path.join("resources", "images", f"{party_of_candidate}", f"{selected_candidate.lower()}.png")):
        #        st.markdown(response)
        #        st.session_state.messages.append({"role": "assistant", "content": response})


        col1, col2 = st.sidebar.columns(2)

        new_chat_button_style = """
        <style>
        .stButton > button {
               background-color: #008CBA; /* Blue color */
            border: none;
            color: white;
            padding: 8px 16px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            }
        </style>
        """

        st.markdown(new_chat_button_style, unsafe_allow_html=True)

        col1, col2 = st.sidebar.columns(2)

        if col1.button("New Chat"):
            st.session_state.session_key = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages = []

        if col2.button("Save Chat"):
            filename = f"chat_{st.session_state.session_key}.csv"

            csv_buffer = StringIO()
            chat_writer = csv.writer(csv_buffer)
            chat_writer.writerow(["Candidate", "Party", "Role", "Question", "Response"])
            question = ""
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    question = msg['content']
                    response = ""  
                elif msg['role'] == 'assistant':
                    response = msg['content']
                    candidate = "Name"  
                    party = get_party(candidate)  
                    chat_writer.writerow([candidate, party, msg['role'], question, response])

            csv_buffer.seek(0)

            st.success(f"Chat saved to {filename}")

            st.download_button(
                label="Download Chat",
                data=csv_buffer.getvalue(),
                file_name=filename,
                mime='text/csv',
                key="download_chat_button"
            )
if __name__ == '__main__':
    main()
