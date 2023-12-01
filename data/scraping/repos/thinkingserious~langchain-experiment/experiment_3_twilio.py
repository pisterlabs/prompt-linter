from langchain.utilities.twilio import TwilioAPIWrapper
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

def create_twilio_api_wrapper(from_number):
    return TwilioAPIWrapper(
        account_sid=os.getenv('TWILIO_ACCOUNT_SID'),
        auth_token=os.getenv('TWILIO_AUTH_TOKEN'),
        from_number=from_number
    )

# Create a new TwilioAPIWrapper object with the from phone number from the environment variables
twilio_sms = create_twilio_api_wrapper(os.getenv('TWILIO_FROM_NUMBER'))

# Use the TwilioAPIWrapper object to send a message with the text "Ahoy!" to the phone number specified in the environment variable
message_sid = twilio_sms.run("Ahoy!", os.getenv('TO_NUMBER'))
print("Message SID:", message_sid)
# You will receive a response containing your Message SID, and an SMS will be sent to you with the text 'Ahoy!'.
# https://support.twilio.com/hc/en-us/articles/223134387-What-is-a-Message-SID-
# >> Message SID: SM...

# Create a new TwilioAPIWrapper object with the WhatsApp from number from the environment variables
twilio_whatsapp = create_twilio_api_wrapper(os.getenv('FROM_WHATSAPP_NUMBER'))

# Use the TwilioAPIWrapper object to send a WhatsApp message with the text "Ahoy!" to the WhatsApp number specified in the environment variable
message_sid = twilio_whatsapp.run("Ahoy!", os.getenv('TO_WHATSAPP_NUMBER'))
print("Message SID:", message_sid)
# You will receive a response containing your Message SID, and a WhatsApp message will be sent to you with the text 'Ahoy!'.
# https://support.twilio.com/hc/en-us/articles/223134387-What-is-a-Message-SID-
# >> Message SID: SM...
