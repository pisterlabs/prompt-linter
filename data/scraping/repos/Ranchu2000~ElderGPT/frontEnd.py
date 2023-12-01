import pytz
import streamlit as st
from streamlit_chat import message
from calendarAgent import load_calendar_chain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
import datetime
import json 
from st_audiorec import st_audiorec
from googleCalendar import list_calendar_events_today
from streamlit_mic_recorder import speech_to_text
from elevenlabs import generate, stream
from elevenlabs import set_api_key, stream, generate, Voices, VoiceSettings, User, Voice
from mainAgent import load_main_agent
import datetime
import os
from dotenv import load_dotenv
load_dotenv()
set_api_key(os.getenv('ELEVENLABS_API_KEY'))

conversationWindow=5 #hardcoded

#st.set_page_config(initial_sidebar_state="collapsed")

## STREAMLIT COMPONENTS
st.title('ElderGPT')
st.subheader(":robot_face: A conversational agent for the elderly")
def reset_memory():
    st.session_state["model_answer_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    memory= ConversationBufferWindowMemory(k=conversationWindow,memory_key="chat_history", return_messages=True)
    st.session_state["memory"]= memory
    
#init session states
if ("chat_answers_history" not in st.session_state 
    and "user_prompt_history" not in st.session_state 
    and "chat_history" not in st.session_state 
    and "memory" not in st.session_state
    and "checkbox" not in st.session_state
    and "contacts" not in st.session_state
    ):
    reset_memory()
    st.session_state["checkbox"]= []
    st.session_state["contacts"]= {}
    

# wav_audio_data = st_audiorec()
# if wav_audio_data is not None:
#     st.audio(wav_audio_data, format='audio/wav') #additional parameters of sample_rate and start time

def on_change_checkbox(date,eventName):
    #fileName= "data/"+ eventName+ ".json"
    if "medication" in eventName.lower():
        fileName= "data/medication.json"
        try:
            with open(fileName, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []
        data.append(date)
        with open(fileName, 'w') as file:
            json.dump(data, file, indent=4)
        st.session_state["checkbox"].append(eventName+date)
    

#side bar for model settings
with st.sidebar:
    st.subheader("Calendar Events")
    events= list_calendar_events_today()
    for event in events:
        time= datetime.datetime.strptime(event[0], '%Y-%m-%dT%H:%M:%S%z').strftime('%I:%M %p')
        pacific = pytz.timezone('America/Los_Angeles')
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        #date= datetime.datetime.strptime(event[0], '%Y-%m-%dT%H:%M:%S%z').strftime('%Y-%m-%d')#-%p for AM or PM
        if event[1]+date not in st.session_state["checkbox"]:
            st.button(label= time+ ": "+event[1], key=event[1], on_click=on_change_checkbox(date,event[1]))
    st.subheader("Configurations")
    with st.expander(":older_man: User Information", expanded=False):
        NAME= st.text_input("Name", value="John Doe")
        EMAIL= st.text_input("Email", value="JohnDoe@gmail.com")
        PHONE= st.text_input("Phone", value="91234567")
        LOCATION= st.text_input("Location", value="2299 Piedmont Ave, Berkeley, CA 94720")
        st.session_state["user_info"]= {"name": NAME, "email": EMAIL, "phone": PHONE, "location": LOCATION}
        # text="My name is {NAME}, my email address is {email}, my contact number is {PHONE} and i stay at {LOCATION}" # not needed- works as it is
        # fText= text.format(NAME=NAME, email=EMAIL, PHONE=PHONE, LOCATION=LOCATION)
        # st.session_state["memory"].save_context({"input":fText}, {"output": "cool. I'll remember it!"})
    with st.expander(":telephone: Contact Information", expanded=False):
        J = st.number_input('Number of Contacts',min_value=0,max_value=10, value=0)
        for j in range(J):
            st.subheader("Contact "+str(j+1))
            NAME= st.text_input("Name", value="John Doe", key="friendsName"+str(j))
            EMAIL= st.text_input("Email", value="JohnDoe@gmail.com", key="friendsEmail"+str(j))
        st.session_state["contacts"][NAME]= EMAIL
    with st.expander(":ear: Audio Settings", expanded=False):
        SPEECH_VOICE= st.selectbox("Voice",options=["Rachel", "Domi", "Bella", "Antoni", "Elli", "Josh", "Jeremy", "Adam", "Sam"])
        READING_RATE = st.number_input('Reading Rate',min_value=0.5,max_value=3.0, value=1.0,step=0.25)
    
    with st.expander(":hammer_and_wrench: Settings ", expanded=False):
        MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','gpt-4','text-davinci-003','text-davinci-002','code-davinci-002'])
        K = st.number_input(' (#) Number of interaction pairs to display',min_value=1,max_value=10, value=3)
        #I= st.number_input(' (#) Number of pairs of conversations to consider',min_value=1,max_value=10) #can't be done as initialisation of buffer memory is on page load


    st.download_button(
        label="Download chat history",
        data= json.dumps(st.session_state["chat_history"]),
        file_name='chat_history_{}.json'.format(datetime.datetime.now()),
        mime='application/json')
    
    st.button(
        label="Clear chat history",
        on_click=reset_memory
    )
    
#Camera
# picture = st.camera_input("Take a picture") #future expansion?
# if picture:
#     st.image(picture)

#upload audio file
#uploaded_file = st.file_uploader("Choose a file") 

#play audio file
# audio_file = open('/Users/yufei/Desktop/Coding/Academics/CS294-GenAI/media/audio.mp3', 'rb')
# audio_bytes = audio_file.read()
# st.audio(audio_bytes, format='audio/mp3')
         
# def run_llm(input_text):
#    qa= load_chain(MODEL)
#    print(st.session_state["memory"])
#    return qa({"question": input_text,"chat_history": st.session_state["chat_history"] })

def run_Calendar(input_text):
   calendarChain= load_calendar_chain(MODEL,st.session_state["memory"], st.session_state["user_info"])
   response= calendarChain.invoke({"input": input_text})
   generated_response={}
   generated_response["answer"]= response["output"] # for backward compatiability
   return generated_response

def run_agent(input_text):
    agent= load_main_agent(MODEL,st.session_state["memory"], st.session_state["user_info"], st.session_state["contacts"])
    response= agent.run(input_text)
    generated_response={}
    generated_response["answer"]= response # for backward compatiability
    return generated_response
    
#act on user's input
def submit():
   with st.spinner("Generating response..."):
    generated_response=run_agent(st.session_state.userPrompt)
    #generated_response, memory= load_chain(query= input_text, model=MODEL)
    st.session_state.user_prompt_history.append(st.session_state.userPrompt)
    st.session_state.model_answer_history.append(generated_response["answer"])
    st.session_state.chat_history.append((st.session_state.userPrompt, generated_response["answer"]))
    st.session_state.memory.save_context({"input": st.session_state.userPrompt},{"output": generated_response["answer"]})
    st.session_state.userPrompt = ""

def speech_to_text_callback():
    if st.session_state.speech_output!= None:
        st.session_state.userPrompt = st.session_state.speech_output
        submit()
    else:
        st.warning('Did not transcribe anything- try to speak again', icon="⚠️")

RFC5646_LANGUAGE_CODES={
    'English': 'en-US',
    'Mandarin': 'cmn-CN',
    'Hindi': 'hi-IN',
    'German': 'de-DE',
    'Japanese': 'ja-JP'
}

c1,c2,c3=st.columns([2,3,5])
with c1:
    st.write("Interact by Speaking:")
with c2:
    language=st.selectbox("Language",options=['English','Mandarin','Hindi','German','Japanese'])
with c3:
    #audio input
    speech_to_text(
        language=RFC5646_LANGUAGE_CODES[language],
        start_prompt="Click to Speak",
        stop_prompt="Click to Stop", 
        just_once=False,
        use_container_width=True,
        callback=speech_to_text_callback,
        args=(),
        kwargs={},
        key="speech"
        )

voiceId= {'Rachel': '21m00Tcm4TlvDq8ikWAM', 'Domi': 'AZnzlk1XvdvUeBnXmlld', 'Bella': 'EXAVITQu4vr4xnSDxMaL', 'Antoni': 'ErXwobaYiN019PkySvjV', 'Elli': 'MF3mGyEYCl7XYWbV9V6O', 'Josh': 'TxGEqnHWrfWFTfGW9XjX', 'Jeremy': 'bVMeCyTHy58xNoL34h3p', 'Adam': 'pNInz6obpgDQGcFmaJgB', 'Sam': 'yoZ06aMxZJJ28mfd3POQ'}
def readOut():
    lastResponse=st.session_state.model_answer_history[-1]
    voice = Voice(
        voice_id=voiceId[SPEECH_VOICE], #Bella
        settings=VoiceSettings(
            stability=0.72, similarity_boost=0.2, style=0.0, use_speaker_boost=False, speaking_rate=READING_RATE
        ),
    )
    audio_stream = generate(
        text=lastResponse,
        voice=voice,
        model="eleven_monolingual_v1",
        stream=True
    )
    stream(audio_stream)
    
col1,col2= st.columns([10,2])
with col1:
    # initial text input
    input_text = st.text_input("User Input", placeholder="Enter your message here...", key="userPrompt", on_change=submit)
with col2:
    st.button("Read aloud", on_click=readOut)
#populate current conversation
with st.expander("Conversation", expanded=True):
    num_items = len(st.session_state.user_prompt_history)
    start_index = max(0, num_items - K)
    if st.session_state.model_answer_history:
        for generated_response, user_query in zip(
            st.session_state.model_answer_history[start_index:],
            st.session_state.user_prompt_history[start_index:],
        ):
            message(
                user_query,
                is_user=True,
            )
            message(generated_response)
