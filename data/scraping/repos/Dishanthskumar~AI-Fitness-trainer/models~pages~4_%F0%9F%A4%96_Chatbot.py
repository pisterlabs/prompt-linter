import openai
import toml
import streamlit as st


def show_messages(text):
    messages_str = [
        f"<span style='color: green;'><b>USER</b>: {_['content']}</span></br>"  if _['role'] == 'user' else f"<span style='color: white;'><b>SYSTEM</b>: {_['content']}</span></br></br>"
        for _ in st.session_state["messages"][1:]
    ]
    text.markdown("Messages", unsafe_allow_html=True)
    text.markdown(str("\n".join(messages_str)), unsafe_allow_html=True)
    
    


with open("../secrets.toml", "r") as f:
    config = toml.load(f)

openai.api_key = "Insert your API key here"

BASE_PROMPT = [{"role": "system", 'content':"""
You are Donnie, an automated Gym assistant to provide workout routines for the users and give suggestions. \
You first greet the customer, then ask them what type of workout routine they want, \
give them a few workout options and wait for them to finalize\ if they ask for changes make those changes accordingly\
, then summarize it and check for a final \
time if the user wants to add anything else. \
If it's a split, you ask for an upper body lower body or back chest and legs split. \

Make sure to clarify all questions about exercises and form \
also make sure to talk only about fitness and fitness related topics\
You respond in a short, very conversational friendly style. \



"""}]

if "messages" not in st.session_state:
    st.session_state["messages"] = BASE_PROMPT

st.header("FIT-BOT")

text = st.empty()
show_messages(text)

if 'something' not in st.session_state:
    st.session_state.something = ''

def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''


st.text_input('Enter message here ', key='widget', on_change=submit)
    # st.write(a)
if st.session_state.something != '':
    with st.spinner("Generating response..."):
        
        st.session_state["messages"] += [{"role": "user", "content": st.session_state.something}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=st.session_state["messages"]
        )
        
        message_response = response["choices"][0]["message"]["content"]
        st.session_state["messages"] += [
            {"role": "system", "content": message_response}
        ]
        show_messages(text)
        
    st.session_state.something = ''

    if st.button("Clear"):
        st.session_state["messages"] = BASE_PROMPT
        show_messages(text)

