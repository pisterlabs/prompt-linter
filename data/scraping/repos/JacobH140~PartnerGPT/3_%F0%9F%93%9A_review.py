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
import gsheet_utils as gs
import logging

# READ: ----------------------------------------------------------------

st.set_page_config(
    page_title="Review",
    page_icon="📚",
)



from streamlit_bokeh_events import streamlit_bokeh_events

from gtts import gTTS
from io import BytesIO
import openai
openai.api_key = api_key

import chat_template

###def get_due_vocab_word(state_object): SUBSUMED BY get_chinese_anki_cards
###    if state_object.simpl_or_trad == "Simplified":
###        field_name = "简体字simplified"
###    else:
###        field_name = "繁体字traditional"
###    deck_name = "中文"
###    due_card_ids = anki_utils.get_due_ids(deck_name=deck_name, limit=1)
###    due_note_info, due_card_info = anki_utils.get_note_and_card_info(due_card_ids)
###    first_key, first_value = next(iter(due_note_info.items()))
###    voc_word = due_note_info[first_key]['fields'][field_name]['value']
###
###    return voc_word

#def get_chinese_anki_cards(state_object, status, limit=None):
#    if state_object.simpl_or_trad == "Simplified":
#        field_name = "简体字simplified"
#    else:
#        field_name = "繁体字traditional"
#    deck_name = "中文"
#
#    if status == "due":
#        card_ids = anki_utils.get_due_ids(deck_name=deck_name, limit=limit)
#    elif status == "new":
#        card_ids = anki_utils.get_new_ids(deck_name=deck_name, limit=limit)
#    elif status == "review":
#        card_ids = anki_utils.get_review_ids(deck_name=deck_name, limit=limit)
#    else:
#        raise ValueError("status must be 'due', 'new', or 'review'")
#    note_info, card_info = anki_utils.get_note_and_card_info(card_ids)
#
#    voc_words = [note_info[key]['fields'][field_name]['value'] for key in note_info.keys()] 
#
#    return voc_words

def learn_get_initial_message(review_state):
    if review_state.custom['out_of_cards']:
        return review_state.messages

    if not review_state.administer_rating_form and f"{review_state.name}_query" not in st.session_state:
        idx = -1
    elif len(review_state.custom['cards_review']) > 1:
        idx = -2
    else:
        idx = -1

    if len(review_state.custom['cards_review']) < 2:
        #review_state.begin_disabled = True
        review_state.custom['out_of_cards'] += 1
    else:
        review_state.begin_disabled = False

  

    print("reviewing: ", review_state.custom['cards_review'])

    card_review = review_state.custom['cards_review'][idx]
    vocab_word = card_review[review_state.simpl_or_trad]




    initial_system = f"""You are a system for dynamic cloze deletion for learning Chinese with Anki: provide with a descriptive sentence at the HSK3 level containing the word {vocab_word}, BUT with the word itself is replaced with blanks. Your sentence should be descriptive, so that it is clear what the correct word is from context. For example, if I were to say "The next word is: 发现", your response might be "我在图书馆 _ _ 了一本有趣的书。" or "我在圖書館 _ _ 了一本有趣的書". Please don't use any English. Then I will probably guess the word. Also, keep track of any grammar mistakes I make if I respond to you in Chinese."""
    initial_user = f"Go!"
    print("initial user: ", initial_user)
    print("vocab_word in LGIM func: ", vocab_word)
    review_state.to_answer = {"text":[review_state.custom['cards_review'][-1][review_state.simpl_or_trad]], "ids": [review_state.custom['cards_review'][-1]["id"]]}
    print("review_state.to_answer in LGIM func: ", review_state.to_answer)

    messages=[
            {"role": "system", "content": initial_system},
            {"role": "user", "content": initial_user},
        ]
    

    return messages

def update_databases(review_state):
    print("retrieving new cards and reviewed cards cache from gsheet")
    from_gsheet_new_cards, wks_new_cards = gs.access_gsheet_by_url_no_df(sheet_name="Due")
    from_gsheet_reviewed_cache, wks_reviewed_cache_cards = gs.access_gsheet_by_url_no_df(sheet_name="Answered Cards Cache")
    excluded = {entry["id"] for entry in from_gsheet_reviewed_cache}
    print("excluded: ", excluded)
    review_state.custom['cards_review'] = [entry for entry in from_gsheet_new_cards if entry["id"] not in excluded] # essentially a set minus
    print("will review: ", review_state.custom['cards_review'])

    from_gsheet_due_cards, wks_due_cards = gs.access_gsheet_by_url_no_df(sheet_name="Due")


def next(review_state):
    print("type: ", type(review_state))
    #if review_state.custom['out_of_cards']:
    #    st.warning("You're out of cards! Head over to **Review** to review some words, or **Converse** to put things into practice...")
    #    return
    update_databases(review_state)



if 'review_state' not in st.session_state:
    st.session_state.review_state = chat_template.SessionNonUIState(name="review_state")

#st.title("Learn")

st.session_state.review_state.next = next
st.session_state.review_state.next_func_args = (st.session_state.review_state,)
    
if not st.session_state.review_state.chatting_has_begun:
    model = st.selectbox("Select a model", ("gpt-3.5-turbo-16k", "gpt-4"))
    st.session_state.review_state.simpl_or_trad = st.selectbox("Simplified or Traditional", ("简体字simplified", "繁体字traditional"))
    st.session_state.review_state.model = model
    

if 'cards_review' not in st.session_state.review_state.custom:
    print("Getting databases (should only see this print once)")
    update_databases(st.session_state.review_state)

if not st.session_state.review_state.custom['cards_review']:
    st.session_state.review_state.begin_disabled = True
    st.warning("There are no cards to review right now! Head over to **Learn** to learn some new words, or **Converse** to put things into practice...")
else:
    st.session_state.review_state.begin_disabled = False


if 'out_of_cards' not in st.session_state.review_state.custom:
    st.session_state.review_state.custom['out_of_cards'] = 0

st.session_state.review_state.to_create_prompt =  """List any words that I asked about or were otherwise new to me during this conversation in the format ['word1', 'word2', ...] and say NOTHING else. Make sure to include the entirety of our conversation."""
st.session_state.review_state.initial_message_func = learn_get_initial_message
st.session_state.review_state.initial_message_func_args = (st.session_state.review_state,)
st.session_state.review_state.end_message = "There are no new words to learn right now! Head over to **Review** to review some words, or **Converse** to put things into practice..."

chat_template.chat(st.session_state.review_state)

if st.session_state.review_state.on_automatic_rerun:
    st.session_state.review_state.on_automatic_rerun = False


logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("hanzipy").setLevel(logging.WARNING)
logging.getLogger("google.auth.transport.requests").setLevel(logging.WARNING)
logging.getLogger("fsevents").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)

