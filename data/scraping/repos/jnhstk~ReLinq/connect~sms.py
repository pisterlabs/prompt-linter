import os
from twilio.rest import Client
import openai
import time
from dotenv import load_dotenv

class SMSLine:
    def __init__(self):
        # Initialize environment variables
        load_dotenv()

        # Initialize Twilio client
        self.twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.twilio_phone_number = os.environ.get("TWILIO_PHONE_NUMBER")
        self.client = Client(self.twilio_account_sid, self.twilio_auth_token)

        # Initialize OpenAI GPT-4 client
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def generate_update_message(self, update, sender_name, recipient_name):
        messages = [
            {"role": "system", "content": "You are a text message generator for Connect.com. \
             Your role is to analyze updates from users' LinkedIn connections and generate \
             personalized, actionable messages. Your responses should be professional, concise, \
             and tailored to the context of each user's connections. BE SURE TO LEAVE A BLANK \
             LINE BEFORE THE POTENTIAL MESSAGE! DO NOT EXCEED 2-3 SENTENCES."},
            {"role": "user", "content" : "sender_name: Mike\n\nrecipient_name: Mike\n\nUpdate: I'm overjoyed to share that \
             I've joined Bonfire as a Product Manager! \
             This is a dream come true, and I'm eager to bring my passion and skills to a team that's \
             doing groundbreaking work. A huge shoutout to everyone who has mentored me, believed in \
             me, and helped me grow professionally and personally. Your support means the world to me. \
             As I embark on this new adventure, I'm looking forward to collaborating with talented \
             individuals and making a meaningful impact. Feel free to reach out for a virtual coffee \
             or a brainstorming session!"},
            {"role": "assistant", "content" : "Hey Mike, Connect.com here!\n\nYour connection Emily Hoggins\
              just started a new role as a Product Manager at Bonfire. Now's a good opporutnity to strenghten\
              your connection! Hey Emily! Congratulations on your new role as a Product Manager at Bonfire!\
              Your journey sounds inspiring, and I'm sure you'll make a significant impact there. Would love \
             to catch up over a virtual coffee soon. Best, Mike"},
            {"role": "user", "content": f"sender_name: {sender_name}\n\nrecipient_name: {recipient_name}\n\n Update: {update}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message['content']

    def generate_tf(self, sender_name, recipient_name, meta=None):
        messages = [
                {"role": "system", "content": "You are a text message generator for Connect.com. \
                    Your role is to generate personalized, actionable messages. Your responses should be professional, concise, \
                    and tailored to the context of each user's connections. BE SURE TO LEAVE A BLANK \
             LINE BEFORE THE POTENTIAL MESSAGE! DO NOT EXCEED 2-3 SENTENCES."},
                {"role": "user", "content": f"recipient_name: Emily\n\n\
                    sender_name: Mike\n\n meta_data: None\n\n"},
                {"role": "assistant", "content": "Hey Mike, reaching out from Connect.com!\
                 Noticed you haven't reached out to emily in 6 months, wanna catch up?\n\n\
                 Hi Emily, \n\nI hope all is well. It's been a while since we last caught up\
                 , and I wanted to see how you're doing. Anything new or exciting happening on your end?\n\n Looking forward to\
                  hearing from you soon. \n\nBest regards, Mike."},
                  {"role": "user", "content": "recipient_name: Felicity\n\nsender_name: Annie\n\nmeta_data:\
                    {'company': 'Airlab', 'duration': '6 months'}\n\n"
                },
                { "role": "assistant",
                    "content": "Hey Annie, it's Connect.com! Noticed it's been 3 months since you last connected with Felicity,\
                          and she's been at Airlab for about 6 months now. How about a message to see how she's settling in?\n\nHi Felicity,\
                              \n\nI hope all is well. I saw that you've been at Airlab for about 6 months now—congratulations! How are you \
                              finding the experience? Anything exciting you're working on?\n\nLooking forward to catching up soon.\n\nBest, Mike."},
                {"role": "user", "content": f"recipient_name: {recipient_name}\n\n\
                 {sender_name}\n\n meta_data: {meta}\n\n"},
            ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message['content']

    def send_sms(self, message, to_phone_number):
        message = self.client.messages.create(
            body=message,
            from_=self.twilio_phone_number,
            to=to_phone_number
        )
        return message.sid

    def process_content(self, message_type, linkedin_update, sender_name, recipient_name, user_phone_number, meta=None):
        if message_type == 'linkedin_update':
            personalized_message = self.generate_update_message(linkedin_update, sender_name, recipient_name)
        elif message_type == 'time_frequency':
            personalized_message = self.generate_tf(sender_name, recipient_name, meta=meta)
        else:
            print("Invalid message type.")
            return

        print(f"Generated message: {personalized_message}")
        sms_status = self.send_sms(personalized_message, user_phone_number)
        print(f"SMS sent with status: {sms_status}")

if __name__ == "__main__":
    line = SMSLine()
    
    # For LinkedIn update
    linkedin_update = open("linkedin_update.txt", "r").read().strip()
    line.process_content(message_type='linkedin_update', linkedin_update=linkedin_update, sender_name="Mike", recipient_name="Emily", user_phone_number="+12403748332")
    # For check-in
    meta_data = {'company': 'Airlab', 'duration': '6 months'}  # Example metadata
    line.process_content(message_type='time_frequency', linkedin_update=None, sender_name="Mike", recipient_name="Sarah", user_phone_number="+12403748332", meta=meta_data)
