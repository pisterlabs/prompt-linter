"""A chatbot that helps the user order food from a restaurant."""

import os

import openai
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from streamlit_chat import message
from streamlit_helpers import generate_response, footer

# Set org ID and API key
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")


def get_system_prompt():
    """Define system prompt for the chatbot."""
    system_prompt = """You are the Waffle House order bot. You are a helpful assistant and will help 
    the customer order their meal. Be friendly and kind at all times. 
    First greet the customer, then collect the order and then ask if it's pick up or delivery. \
    You wait to collect the entire order, then summarize it and check for a final time if the \
    customer wants to add anything else. \
    Always summarize the entire order before collecting payment. \
    If it's a delivery, ask them for their address. \
    If it's pick up, tell them our address: 123 Waffle House Lane, London. \
    Finally collect the payment. Ask if they want to pay by credit card or cash. \
    If they say credit card say 'Please click the link below to pay by credit card'. \
    If they say cash, say they can pay when they pick up the order or pay the delivery driver. \
    Make sure to clarify all options, extras and sizes to uniquely identify the order. \
    The menu is: \
    Waffle type: normal ($10), gluten-free ($10), protein ($1 extra) \
    Toppings: strawberries, blueberries, chocolate chips, whipped cream, butter, syrup, bacon \
    Each topping costs $1 \
    Drinks: coffee, orange juice, milk, water \
    Each drink costs $2 \
    Once the order is complete, output the order summary and total cost in JSON format. \
    Itemize the price for each item. The fields should be 1) waffle_type, 2) list of toppings \
    3) list of drinks, 4) total price (float)
    """
    system_prompt = system_prompt.replace("\n", " ")
    return system_prompt


# Top matter
st.set_page_config(page_title="Waffle House Order Bot", page_icon=":waffle:")
st.title("Waffle House Order Bot 🧇")

intro = """Welcome to the Waffle House, the place where all your waffle filled dreams come true.

Start chatting with WaffleBot below to find out what you can order, how much it costs, and how to pay."""
st.markdown(intro)

# Add footer
source_link = "https://github.com/codeananda/ChatGPT_Projects/blob/main/waffle_bot.py"
footer(source_link)

initial_state = [
    {"role": "system", "content": get_system_prompt()},
    {"role": "assistant", "content": "👋 Welcome to Waffle House! What can I get for you?"},
]

if "messages" not in st.session_state:
    st.session_state["messages"] = initial_state

# Let user clear the current conversation
clear_button = st.sidebar.button("Clear Conversation", key="clear")
if clear_button:
    st.session_state["messages"] = initial_state

# Chat history container
response_container = st.container()
# Text input container
input_container = st.container()

with input_container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("You:", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output = generate_response(user_input)

if st.session_state["messages"]:
    with response_container:
        for message_ in st.session_state.messages:
            if message_["role"] == "user":
                message(message_["content"], is_user=True)
            elif message_["role"] == "assistant":
                message(message_["content"], avatar_style="thumbs")
