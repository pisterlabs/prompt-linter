import datetime
import hubspot
import langchain
import openai
import streamlit

# set follow-up message timeframe in days
FOLLOWUP_DELAY = 7

def send_followup_message(customer_email):
    """
    Sends a follow-up message to the customer using Langchain and OpenAI.
    """

    # retrieve customer information from Hubspot
    customer_info = hubspot.get_customer_info(customer_email)

    # generate personalized message using Langchain
    message = langchain.generate_message(customer_info)

    # send message using OpenAI's GPT-3
    openai.send_message(customer_email, message)

# check for customers who need a follow-up message
for customer_email in hubspot.get_unresponsive_customers():
    # calculate delay since last message
    last_message_date = hubspot.get_last_message_date(customer_email)
    delay = datetime.datetime.now() - last_message_date

    if delay.days >= FOLLOWUP_DELAY:
        send_followup_message(customer_email)
