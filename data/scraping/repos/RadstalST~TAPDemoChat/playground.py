import streamlit as st
from langchain.prompts import PromptTemplate
from agents import PlaygroundBot
import io
import json 
import datetime
from agents import chatHistory

# initialize exportDict in session state
if  "exportDict" not in st.session_state:
    st.session_state.exportDict = {
        "mode":"",
        "userInput":"",
        "prompt":"",
        "response":"",
        "rawResponse":"",
        "feedback":"",
        "timestamp":""
        }
  
if "modeIndex" not in st.session_state:
    st.session_state.modeIndex = 0

def setTimeStamp():
    st.session_state.exportDict["timestamp"] = datetime.datetime.now()

@st.cache_resource
def playGroundBotSelector(option:str)->PlaygroundBot.BasePlaygroundBot:
    
    match option:
        case "GPT4":
            return PlaygroundBot.PlayGroundGPT4()
        case "GPT4+ToT":
            return PlaygroundBot.PlayGroundGPT4ToT()
        case "GPT4+CoT":
            return PlaygroundBot.PlayGroundGPT4CoT()
        case "GPT+CoT+Chroma":
            return PlaygroundBot.PlayGroundGPT4CoTChroma()
        case _:
            return PlaygroundBot.BasePlaygroundBot()     



st.header("Welcome to Playground")
st.write("This is a demo of the Med Bot")
st.warning("if you are editing the code in modules, please restart the app or press 'c' (clear resource cache) to see the changes")
questionPane = st.container()
st.divider()
formPane = st.container()
resultContainer = st.container()
feedbackPane = st.container()

# container for history
history_container = st.sidebar.container()

# creating an empty list to store conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []


if getattr(st.session_state, 'status', None) is None:
    st.session_state['status'] = {}
else:
    st.session_state.status = {}


prompt_template = PromptTemplate.from_template(
                    """
                    \nPrompt: {prompt}
                    \nUser Input: {userInput}
                    """
                ) 
generated_prompt = ""
playgroundbot = PlaygroundBot.BasePlaygroundBot() # empty model


with questionPane:
    final_prompt = ""
    with st.container() as form:
        option = st.selectbox("bot option",('GPT4', 'GPT4+ToT', 'GPT4+CoT',"GPT+CoT+Chroma"),index=st.session_state.modeIndex)
        st.session_state.exportDict["mode"] = option
        playgroundbot = playGroundBotSelector(option)
               

        with st.expander("description",expanded=True):
            st.write(playgroundbot.getDescription())
    
        col1, col2 = st.columns(2)
        with col1: 
            st.subheader("User Input")
            userInput = st.text_area(
                "Your input goes here:", 
                placeholder="I have problem with headache today. I worked 10 hours yesterday",
                value="I have problem with headache today. I worked 10 hours yesterday",
                key='input',
                height=300)
        with col2:
            st.subheader("Scenario")
            prompt = st.text_area(
                "Your prompt goes here:", 
                key='prompt', height=300, 
                value="Please provide possible symptom with my problem")
            

        if prompt:
            final_prompt = prompt
        else:
            final_prompt = option
        with st.expander("See Generated Prompt"):
            
            generated_prompt = prompt_template.format(userInput=userInput, prompt=final_prompt)
            
            # st.write(form.__dict__)
with formPane:        
    with st.form("playground_form"):
        st.markdown(generated_prompt)

        submit_button = st.form_submit_button(label='Send')
        if submit_button:
            with st.status('Wait for it...',expanded=True):
                st.session_state.exportDict["userInput"] = userInput # save to export dict
                st.session_state.exportDict["prompt"] = prompt # save to export dict
                
                st.write("getting response from the bot")
                result = playgroundbot.ask(generated_prompt)
                st.session_state.exportDict["response"] = result["response"] # save to export dict
                st.session_state.exportDict["rawResponse"] = result
            resultContainer.subheader("Bot Response")
            playgroundbot.display(resultContainer,st.session_state.exportDict["rawResponse"])

            with resultContainer.expander("debug"):
                st.write(result)

            # adding the user input and bot response to the conversation history
            user_input = final_prompt
            response = chatHistory.add_user_input_to_history(user_input, result)

            # Display the updated conversation history in the sidebar
            with history_container:
                st.subheader("Conversation History")
                for entry in st.session_state.conversation_history:
                    st.write(entry[0])

with feedbackPane:

    # feedback = st.text_area("your feedback:", key='feedback', height=50,placeholder="please input feedback with 50 character or more")
    # st.session_state.exportDict["feedback"] = feedback

    st.write("---")
    st.header("We would love to hear from you!")
    st.write("##")

    # Refer: https://formsubmit.co/
    feedback_form = """
    <form action="https://formsubmit.co/6d5189f5e008a3398f3c9b2bfee1a576" method="POST" target="_blank">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Name" required>
        <input type="email" name="email" placeholder="Email" required>
        <textarea name="message" placeholder="Write your feedback here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """

    st.markdown(feedback_form, unsafe_allow_html=True)

    st.write("---")

# Use custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style/style.css")

st.download_button(
    "Download interaction",
    json.dumps(st.session_state.exportDict, indent=4, sort_keys=True, default=str),
    file_name="interaction.json",
    mime="application/json",
    # disabled=(feedback == "" or len(feedback)<=50),
    on_click=setTimeStamp
    )



