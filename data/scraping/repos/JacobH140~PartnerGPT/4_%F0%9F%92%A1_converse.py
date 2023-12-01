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




from streamlit_bokeh_events import streamlit_bokeh_events

from gtts import gTTS
from io import BytesIO
import openai
openai.api_key = api_key

import chat_template

st.set_page_config(
    page_title="Converse",
    page_icon="💡",
)

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

def learn_get_initial_message(converse_state):


    if not converse_state.administer_rating_form and f"{converse_state.name}_query" not in st.session_state:
        idx = -1
    elif len(converse_state.custom['cards_converse']) > 1:
        idx = -2
    else:
        idx = -1

  

  

    print("reviewing: ", converse_state.custom['cards_converse'])

    if st.session_state.converse_state.custom['topic'] == "Surprise Me!":
        # choose topic with greatest number of relevant cards
        topics = ['animals and wildlife', 'art and literature', 'Astronomy and Space', 'business and finance', 'Career and Professional Development', 'cars and transportation', 'Celebrity and Pop Culture', 'Chinese Mythology and Folklore', 'Cooking and Cuisine', 'culture and customs', 'daily routines', 'Dating and Relationships', 'education', 'Elderly Care and Retirement', 'entertainment', 'environment and sustainability', 'Etiquette and Social Norms', 'family and relationships', 'fashion and style', 'food and drink', 'Gardening and Plants', 'geography and landmarks', 'health and wellness', 'history and traditions', 'Hobbies and Interests', 'holidays and celebrations', 'Home and Lifestyle', 'Human Rights and Social Issues', 'language and linguistics', 'learning chinese', 'Marriage and Weddings', 'math, science and innovation', 'Mental Health and Wellness', 'Military and Defense', 'movies and television', 'music and audio', 'Outdoor Activities and Adventures', 'Parenting and Childcare', 'Personal Growth and Self-Improvement', 'Pets and Pet Care', 'philosophy and religion', 'Photography and Visual Arts', 'politics and current events', 'Public Transport and Infrastructure', 'Real Estate and Housing', 'science and technology', 'shopping', 'Social Media and Internet Culture', 'sports and fitness', 'technology', 'travel', 'travel and tourism', 'Volunteering and Community Service', 'weather and climate', 'work and employment']
        for i in range(len(topics)):
            topics[i] = topics[i].replace(" ", "_")
        scores = [0]*len(topics)
        for topic in topics:
            for c in converse_state.custom['cards_converse']: 
                print("c: has tags", c['tags'])
                print("topic is:", topic)
                print("topic in c['tags'] is: ", topic in c['tags'])
                if topic in c['tags']:
                    print("topic: ", topic, " is in the tags of ", c)
                    scores[topics.index(topic)] += 1
        if scores == [0]*len(topics):
            st.session_state.converse_state.custom['topic'] = topics[randrange(len(topics))]
        else:
            st.session_state.converse_state.custom['topic'] = topics[scores.index(max(scores))]
        print("topics: ", topics)
        print("scores: ", scores)
    print("topic: ", st.session_state.converse_state.custom['topic'])
                    
            
    topic_cards = [c for c in converse_state.custom['cards_converse'] if st.session_state.converse_state.custom['topic'] in c["tags"]]


    # if there are less than 20 topic cards, fill in the rest with random cards
    if len(topic_cards) < 20:
        topic_cards += [c for c in converse_state.custom['cards_converse'] if c not in topic_cards][:20-len(topic_cards)]
    
    #print("TOPIC CARD", topic_cards[0])
    words = [c[converse_state.simpl_or_trad] for c in topic_cards]

    

    initial_system = f"""You are a Chinese language partner (Difficulty: HSK3), and we are going to discuss {st.session_state.converse_state.custom['topic']}. Please incorporate the following words into our conversation: {words}."""
    initial_user = f"Go! Remember to, as we go, discuss the words {words}. You should **bolden** them in your messages."
    print("initial user: ", initial_user)
    converse_state.to_answer = {"text": [c[converse_state.simpl_or_trad] for c in topic_cards], "ids": [c["id"] for c in topic_cards]}
    print("converse_state.to_answer in LGIM func: ", converse_state.to_answer)

    messages=[
            {"role": "system", "content": initial_system},
            {"role": "user", "content": initial_user},
        ]
    

    return messages

def update_databases(converse_state):
    print("retrieving new cards and reviewed cards cache from gsheet")
    from_gsheet_new_cards, wks_new_cards = gs.access_gsheet_by_url_no_df(sheet_name="Due")
    from_gsheet_reviewed_cache, wks_reviewed_cache_cards = gs.access_gsheet_by_url_no_df(sheet_name="Answered Cards Cache")
    excluded = {entry["id"] for entry in from_gsheet_reviewed_cache}
    print("excluded: ", excluded)
    converse_state.custom['cards_converse'] = [entry for entry in from_gsheet_new_cards if entry["id"] not in excluded] # essentially a set minus
    print("will review: ", converse_state.custom['cards_converse'])

    from_gsheet_due_cards, wks_due_cards = gs.access_gsheet_by_url_no_df(sheet_name="Due")


def next(converse_state):
    print("type: ", type(converse_state))
    #if converse_state.custom['out_of_cards']:
    #    st.warning("You're out of cards! Head over to **Review** to review some words, or **Converse** to put things into practice...")
    #    return
    update_databases(converse_state)



if 'converse_state' not in st.session_state:
    st.session_state.converse_state = chat_template.SessionNonUIState(name="converse_state")

#st.title("Learn")

st.session_state.converse_state.next = next
st.session_state.converse_state.next_func_args = (st.session_state.converse_state,)
    
if not st.session_state.converse_state.chatting_has_begun:
    model = st.selectbox("Select a model", ("gpt-3.5-turbo-16k", "gpt-4"))
    st.session_state.converse_state.simpl_or_trad = st.selectbox("Simplified or Traditional", ("简体字simplified", "繁体字traditional"))
    st.session_state.converse_state.model = model
    st.session_state.converse_state.custom['topic'] =  st.selectbox("Topic (Optional)", ("Surprise Me!", "I'll Start", 'animals and wildlife', 'art and literature', 'Astronomy and Space', 'business and finance', 'Career and Professional Development', 'cars and transportation', 'Celebrity and Pop Culture', 'Chinese Mythology and Folklore', 'Cooking and Cuisine', 'culture and customs', 'daily routines', 'Dating and Relationships', 'education', 'Elderly Care and Retirement', 'entertainment', 'environment and sustainability', 'Etiquette and Social Norms', 'family and relationships', 'fashion and style', 'food and drink', 'Gardening and Plants', 'geography and landmarks', 'health and wellness', 'history and traditions', 'Hobbies and Interests', 'holidays and celebrations', 'Home and Lifestyle', 'Human Rights and Social Issues', 'language and linguistics', 'learning chinese', 'Marriage and Weddings', 'math, science and innovation', 'Mental Health and Wellness', 'Military and Defense', 'movies and television', 'music and audio', 'Outdoor Activities and Adventures', 'Parenting and Childcare', 'Personal Growth and Self-Improvement', 'Pets and Pet Care', 'philosophy and religion', 'Photography and Visual Arts', 'politics and current events', 'Public Transport and Infrastructure', 'Real Estate and Housing', 'science and technology', 'shopping', 'Social Media and Internet Culture', 'sports and fitness', 'technology', 'travel', 'travel and tourism', 'Volunteering and Community Service', 'weather and climate', 'work and employment',))
    

if 'cards_converse' not in st.session_state.converse_state.custom:
    print("Getting databases (should only see this print once)")
    update_databases(st.session_state.converse_state)







st.session_state.converse_state.to_create_prompt =  """List any words that I asked about or were otherwise new to me during this conversation in the format ['word1', 'word2', ...] and say NOTHING else. Make sure to include the entirety of our conversation."""
st.session_state.converse_state.initial_message_func = learn_get_initial_message
st.session_state.converse_state.initial_message_func_args = (st.session_state.converse_state,)

chat_template.chat(st.session_state.converse_state)

if st.session_state.converse_state.on_automatic_rerun:
    st.session_state.converse_state.on_automatic_rerun = False


logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("hanzipy").setLevel(logging.WARNING)
logging.getLogger("google.auth.transport.requests").setLevel(logging.WARNING)
logging.getLogger("fsevents").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)

