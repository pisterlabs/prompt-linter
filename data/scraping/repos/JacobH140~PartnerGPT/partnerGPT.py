import streamlit as st
from streamlit_chat import message
from utils import get_initial_message, get_chatgpt_response, get_chatgpt_response_stream_chunk, update_chat, stream_chat_completion
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
import string
from playsound import playsound
import base64
import chinese_nlp_utils as cnlp



from streamlit_bokeh_events import streamlit_bokeh_events

from gtts import gTTS
from io import BytesIO
import openai
openai.api_key = api_key

def remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
                md,
                unsafe_allow_html=True,
            )

punctuation = r"""!?.;:。。()[]？。、；：-""" 
def has_punctuation(s):  
    return any([c in punctuation for c in s])

def remove_spaces_punctuation(input_string):
    # Remove spaces
    #input_string_no_space_punc = input_string.replace(" ", "")
    input_string_no_space_punc = input_string
    for c in input_string_no_space_punc:
        if c in punctuation:
            input_string_no_space_punc = input_string_no_space_punc.replace(c, "")
    
    # Remove punctuation
    #input_string = input_string.translate(str.maketrans("", "", string.punctuation))
    print("BEFORE ", input_string)
    input_string_no_space_punc = ' '.join(input_string_no_space_punc.split())
    print("AFTER ", input_string_no_space_punc)
    return input_string_no_space_punc

class SessionNonUIState:
    def __init__(self):
        self.vocab_words_testing_temp = ["滑冰", "大象", "默契", "矛盾"]
        self.aux_words_testing_temp = ["熊猫", "熊猫", "影响"]

        self.audio_playing = False

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

    def user_text_input_widget(self, tr, session_state_object, value=None):
        if value is None:
            value = session_state_object['query'] 
            #tr.text_input("You: ", value=session_state_object['query'], key="query", placeholder='speak or type', label_visibility="collapsed", on_change=clear_text, disabled=self.administer_rating_form)
        #else:
        tr.text_input("You: ", key="query", value="test value", placeholder='speak or type', label_visibility="collapsed", on_change=clear_text, disabled=self.administer_rating_form)


    def stream_response(self, messages, real_time_audio=False):
        # real-time audio is experimental and will only work when running locally from mac.
        with st.empty():
            # create variables to collect the stream of chunks
            collected_chunks = []
            collected_messages = []
            # iterate through the stream of events
            for chunk in stream_chat_completion(messages, self.model, temperature=0.5):
                collected_chunks.append(chunk)  # save the event response
                chunk_message = chunk['choices'][0]['delta']  # extract the message
                collected_messages.append(chunk_message)  # save the message
                response = ''.join([m.get('content', '') for m in collected_messages])
                prev_response_so_far = st.session_state['response_so_far']
                if len(response) > len(prev_response_so_far):
                   st.success(response)
                   print("new response_so_far", response)
                   print("old response_so_far", prev_response_so_far)
                   print("sonified so far", st.session_state['sonified_so_far'])
                   to_sonify = response[len(st.session_state['sonified_so_far']):len(response)]
                   v = response[len(prev_response_so_far):len(response)]      
                   if has_punctuation(v) and real_time_audio: # only sonify clause-like structures, and  don't sonify parentheses contents 
                        split, langs = cnlp.split_text(to_sonify)
                        print("--ENTERED SONIFICATION STAGE, split is-- ", split)
                        for s in range(len(split)):
                            print('candidate string is ', split[s])
                            print("BEFORE PINYIN/PUNC REMOVAL", split[s])
                            print("AFTER PUNC REMOVAL", remove_spaces_punctuation(split[s]))
                            print("AFTER PINYIN/PUNC REMOVAL", cnlp.remove_pinyin_tone_marked_ish(remove_spaces_punctuation(split[s])))
                            if cnlp.remove_pinyin_tone_marked_ish(remove_spaces_punctuation(split[s])):
                                print("SONIFYING: ", split[s])
                                tts = gTTS(cnlp.remove_pinyin_tone_marked_ish(remove_spaces_punctuation(split[s])).strip(), lang=langs[s])
                                tts.save("sonify" + '.mp3')
                                #if not cnlp.is_pinyin_tone_marked(remove_spaces_punctuation(split[s])):
                                playsound("sonify" + '.mp3')
                                print("sonifying: ", to_sonify)
                                st.session_state['sonified_so_far'] = response
                       #st.audio(to_sonify + '.mp3')
                   st.session_state['response_so_far'] = response
                       #st.audio(to_sonify + '.mp3')
                   st.session_state['currently_sonifying'] = to_sonify
                   st.success(response)
                       #autoplay_audio(st.session_state['currently_sonifying'] + '.mp3')
                       #time.sleep(3) 
        
                st.success(response)
            if not real_time_audio and not self.audio_playing:
                pass
                ##tts = gTTS(remove_text_inside_brackets(response), lang='zh-cn')
                ##print(remove_text_inside_brackets(response))
                ##tts.save("response.mp3")
                ##autoplay_audio("response.mp3")
                ###st.success(response)
                ###update_UI_messages(self)

            if st.session_state['response_so_far'] == st.session_state['sonified_so_far']:
                st.session_state['response_so_far'] = ""
                st.session_state['sonified_so_far'] = ""

            return response

    def generate_bot_response_placeholder(self, query):
        #with st.spinner("generating..."):
        messages = self.messages
        messages = update_chat(messages, "user", query)
        self.past.append(query)
        update_UI_messages(self)
        response = self.stream_response(messages, real_time_audio=False) # this is where the model would be called etc
        messages = update_chat(messages, "assistant", response)
        self.generated.append(response)
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
    for i in range(len(state_object.past)): # reverse iterate through list
        #st.info(state_object.past[i]) if state_object.past[i] else None # user messages
        #st.success(state_object.generated[i]) if state_object.generated[i] else None # bot responses
        #st.warning(state_object.reviewed[i]) if state_object.reviewed[i] else None # review notification for when the 'next' form is submitted

        # rewrite above using regular if statements, not as comment
        if state_object.past[i]:
            st.info(state_object.past[i])
        if i < len(state_object.generated) and state_object.generated[i]:
            st.success(state_object.generated[i])
        if i < len(state_object.generated) and state_object.reviewed[i]:
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

def UI_controls(nonUI_state): 
    st.divider()
    mic, user, next_button = st.columns([2,30,4])
    with mic:
        st.button("🎙️", key="mic_button", disabled=nonUI_state.administer_rating_form)
        #if 'stt_session' not in st.session_state:
        #    st.session_state['stt_session'] = 0 # init
        #stt_button = stt.mic_button()
    with user:
        if 'query' not in st.session_state:
            st.session_state['query'] = ''
        tr = st.empty()
        nonUI_state.user_text_input_widget(tr, st.session_state)
        #stt.mic_button_monitor(tr, nonUI_state, stt_button, st.session_state) 
        
    with next_button:
        st.button("Next", key="next_button", on_click=on_proceed_button_click, disabled=nonUI_state.administer_rating_form)

def on_proceed_button_click():
    nonUI_state.messages = get_initial_message(nonUI_state.vocab_words_testing_temp.pop(), nonUI_state.aux_words_testing_temp)
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


def chat(nonUI_state):
    if "queried" not in st.session_state:
        st.session_state["queried"] = ""

    if 'response_so_far' not in st.session_state:
        st.session_state['response_so_far'] = ""

    if 'sonified_so_far' not in st.session_state:
        st.session_state['sonified_so_far'] = ""
    
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
        #update_UI_messages(nonUI_state)
        

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
        UI_controls(nonUI_state)

                

    
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