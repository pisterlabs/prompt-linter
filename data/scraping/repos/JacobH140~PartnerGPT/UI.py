import streamlit as st
from streamlit_chat import message
from utils import get_initial_message, get_chatgpt_response, get_chatgpt_response_stream_chunk, update_chat
import os
from dotenv import load_dotenv
import openai
from secret_openai_apikey import api_key
import anki_utils
from collections import defaultdict
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from random import randrange
import stt



from streamlit_bokeh_events import streamlit_bokeh_events

from gtts import gTTS
from io import BytesIO
import openai
os.environ['OPENAI_API_KEY'] = api_key 
load_dotenv()

class SessionNonUIState:
    def __init__(self):
        self.chatting_has_begun = False
        self.model = None
        self.messages = []

        self.generated = []
        self.past = []
        self.reviewed = []

        self.submitted_ratings = defaultdict(lambda: None)
        self.submitted_rating =  defaultdict(lambda: None)
        self.administer_rating_form = False
        self.form_submit_button_clicked = False
        self.on_automatic_rerun = False # this is specific to the form thing... don't try to use it for anything else

    def user_text_input_widget(self, session_state_object, value=None):

        
        if 'query' not in session_state_object:
            st.text_input("You: ", placeholder='speak or type', key="query", label_visibility="collapsed", on_change=clear_text, disabled=self.administer_rating_form)
        else:
            if value is None:
                st.text_input("You: ", value=session_state_object['query'], placeholder='speak or type', key="query", label_visibility="collapsed", on_change=clear_text, disabled=self.administer_rating_form)
            else:
                st.text_input("You: ", value=value, placeholder='speak or type', key="query", label_visibility="collapsed", on_change=clear_text, disabled=self.administer_rating_form)




    def generate_bot_response_placeholder(self, query):
        with st.spinner("generating..."):
            messages = self.messages
            messages = update_chat(messages, "user", query)
            response = "generated response placeholder for query " + query # this is where the model would be called etc
            messages = update_chat(messages, "assistant", response)
            self.generated.append(response)
            self.past.append(query)
            self.reviewed.append("")


        return messages
    
    def review_notif(self, str):
        self.generated.append("")
        self.past.append("")
        self.reviewed.append(str)

    def reset_submitted_ratings(self):
        self.submitted_ratings = defaultdict(lambda: None)
        assert(len(self.submitted_ratings) == 0)

    def rerun(self):
        self.on_automatic_rerun = True
        st.experimental_rerun()

def get_initial_message_placeholder():
    messages=[
            {"role": "system", "content": "system initial message placeholder"},
            {"role": "user", "content": "user initial message placeholder"},
            {"role": "assistant", "content": "assistant initial message placeholder"},
        ]
    return messages


    
 
def update_UI_messages(state_object):
    for i in range(len(state_object.generated)): # reverse iterate through list
        #st.info(state_object.past[i]) if state_object.past[i] else None # user messages
        #st.success(state_object.generated[i]) if state_object.generated[i] else None # bot responses
        #st.warning(state_object.reviewed[i]) if state_object.reviewed[i] else None # review notification for when the 'next' form is submitted

        # rewrite above using regular if statements, not as comment
        if state_object.past[i]:
            st.info(state_object.past[i])
        if state_object.generated[i]:
            st.success(state_object.generated[i])
        if state_object.reviewed[i]:
            st.warning(state_object.reviewed[i])



def initialize_app(heading, subheading):
    #st.title(heading)
    #st.subheader(subheading)
    if 'state' not in st.session_state:
        st.session_state.state = SessionNonUIState()

    if not st.session_state.state.chatting_has_begun:
        model = st.selectbox("Select a model", ("gpt-3.5-turbo", "gpt-4"))
        st.session_state.state.model = model
    
    return st.session_state.state

def expander_messages_widget(state_object):
    with st.sidebar.expander("Show Current Messages"):
        st.write(state_object.messages)
        

def rating_form(nonUI_state):
    def on_submit():
        nonUI_state.administer_rating_form = False
        nonUI_state.form_submit_button_clicked = True
        st.balloons()

    with st.form('Rating Form', clear_on_submit=False):
        ratings = defaultdict(lambda: None)
        #for word in [state_object.main_words[-2]] + state_object.current_aux_words:
        for word in ["word1", "word2", "word3", "word4"]:
            r = st.radio(word, ("Again", "Hard", "Good", "Easy", "N/A"), horizontal=True)
            #r =  st.text_input(word) # also yields all blanks
            ratings[word] = r
        #val = {randrange(100):st.slider("slider")}
        return st.form_submit_button('Submit', on_click=on_submit), ratings

def ratings_widget(state_object):
    with st.sidebar.expander("Show Review Ratings"):
        st.write(state_object.submitted_ratings)


def clear_text(): # should this have st.session_state as an argument?
    st.session_state["queried"] = st.session_state["query"]
    st.session_state["query"] = ""

def chat(nonUI_state):



    def on_proceed_button_click():
        nonUI_state.messages = get_initial_message_placeholder()
        if nonUI_state.chatting_has_begun:
            nonUI_state.administer_rating_form = True
            clear_text()
        
        if 'query' in st.session_state and not nonUI_state.administer_rating_form: # first part ensures this doesn't run when 'begin' is pressed, second part ensures 'next' behavior doesn't run while the rating form is being administered
            st.session_state.queried = '**next !**'
            nonUI_state.messages = nonUI_state.generate_bot_response_placeholder(query=st.session_state.queried)
            clear_text()
        elif not nonUI_state.administer_rating_form:
            nonUI_state.messages = nonUI_state.generate_bot_response_placeholder(query='**begin !**')
            #st.markdown("<details> <summary>Details</summary> Something small enough to escape casual notice. </details>", unsafe_allow_html=True)
        
        nonUI_state.chatting_has_begun = True
        
        

    if "queried" not in st.session_state:
        st.session_state["queried"] = ""
    
    #st.button("Begin" if not nonUI_state.chatting_has_begun else "Next", key="proceed_button", on_click=on_proceed_button_click)
    if not nonUI_state.chatting_has_begun:
        st.button("Begin", key="begin_button", on_click=on_proceed_button_click)


    if 'query' in st.session_state and st.session_state.queried:
        nonUI_state.messages = nonUI_state.generate_bot_response_placeholder(st.session_state.queried)

    if nonUI_state.generated:
        #if nonUI_state.submitted_ratings and not nonUI_state.on_automatic_rerun:
        #    nonUI_state.review_notif("**Submitted ratings for <num_items> items: <items>**")
        #    print(nonUI_state.submitted_ratings)
            #if nonUI_state.on_automatic_rerun:
                #nonUI_state.reset_submitted_ratings()
        #print("rerun status", nonUI_state.on_automatic_rerun)
        #print("query in session state? ", 'query' in st.session_state)
        
        if nonUI_state.on_automatic_rerun: # first part ensures this doesn't run when 'begin' is pressed, second part ensures 'next' behavior doesn't run while the rating form is being administered
            st.session_state.queried = '**next !**'
            #print("queried:", st.session_state.queried)
            nonUI_state.review_notif(f"**Submitted ratings for <num_items> items: <items>**")
            nonUI_state.messages = nonUI_state.generate_bot_response_placeholder(query=st.session_state.queried)
            

        #print(nonUI_state.messages)
        update_UI_messages(nonUI_state)

        #if nonUI_state.form_submit_button_clicked:
        #    st.warning("**Submitted ratings for <num_items> items: <items>**") # not ideal, but this requires least amount of state management

        
    
        

    #print("administer_rating_form:", nonUI_state.administer_rating_form, "", "form_submit_button_clicked:", nonUI_state.form_submit_button_clicked, "", "on_automatic_rerun:", nonUI_state.on_automatic_rerun)
    if nonUI_state.administer_rating_form or nonUI_state.form_submit_button_clicked: #nonUI_state.on_automatic_rerun:
        submitted_form, nonUI_state.submitted_rating = rating_form(nonUI_state) # only want to call rating form when administer_rating_form is True, but only care about the results when it's false...?
        #print(submitted_form, nonUI_state.submitted_rating)
        if nonUI_state.form_submit_button_clicked and not nonUI_state.on_automatic_rerun: # note that at all times AT MOST one of form_submit_button_clicked, administer_rating_form, and on_automatic_rerun can be True
         #   print("rerun")
            nonUI_state.form_submit_button_clicked = False
            nonUI_state.rerun()
        #nonUI_state.administered_rating_form = True

            
    if nonUI_state.on_automatic_rerun: # this is when the submitted ratings are the real ones
        #print("val(s):", nonUI_state.submitted_rating)
        nonUI_state.submitted_ratings = nonUI_state.submitted_ratings | nonUI_state.submitted_rating # dictionary merge
        #nonUI_state.administered_rating_form = False
        #nonUI_state.reset_submitted_ratings()
        #nonUI_state.reviewed.append("**Submitted reviews for <num_items> items: <items> !**")
        #notification = "**Submitted ratings for <num_items> items: <items>**"
        
        #print(nonUI_state.submitted_ratings)
        #if nonUI_state.on_automatic_rerun:
            #nonUI_state.reset_submitted_ratings()
    


    
    if nonUI_state.chatting_has_begun:
        st.divider()
        mic, user, next_button = st.columns([2,30,4])
        with mic:
            #st.button("🎙️", key="mic_button", disabled=nonUI_state.administer_rating_form)
            if 'query' not in st.session_state:
                st.session_state['stt_session'] = 0 # init
            mic_result = stt.mic_button(st.session_state)
        with user:
            stt.mic_button_monitor(nonUI_state, mic_result, st.session_state) # first time query appears
            
        with next_button:
            st.button("Next", key="next_button", on_click=on_proceed_button_click, disabled=nonUI_state.administer_rating_form)

                

    
    if nonUI_state.generated:
        expander_messages_widget(nonUI_state)
        ratings_widget(nonUI_state)



if __name__ == '__main__':
    nonUI_state = initialize_app("header", "subheader")
    chat(nonUI_state)
    if nonUI_state.on_automatic_rerun:
        nonUI_state.on_automatic_rerun = False
    #def form_func():
    #    def on_submit(val):
    #        nonUI_state.administer_rating_form = False
    #        nonUI_state.submitted_ratings = {"val":val}
    #        st.balloons()
    #    with st.form('Rating Form', clear_on_submit=False):
    #        #for word in [state_object.main_words[-2]] + state_object.current_aux_words:
    #        #for word in ["word1", "word2", "word3", "word4"]:
    #            #r = st.radio(word, ("Again", "Hard", "Good", "Easy", "N/A"), horizontal=True)
    #            #r =  st.text_input(word) # also yields all blanks
    #            #nonUI_state.submitted_ratings[word] = r
    #        val = st.slider("slider")
    #        return st.form_submit_button('Submit', on_click=on_submit, args=(val,)), val
    #        
    #if nonUI_state.administer_rating_form: 
    #    nonUI_state.submitted_form, nonUI_state.submitted_ratings = form_func()
    #if nonUI_state.submitted_form:
    #    print("val(s):", nonUI_state.submitted_ratings)
    #    nonUI_state.submitted_form = False


#https://stackoverflow.com/questions/52718897/minimal-shortest-html-for-clickable-show-hide-text-or-spoiler
