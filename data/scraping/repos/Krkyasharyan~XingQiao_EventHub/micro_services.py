import smtplib
import os
import openai
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

SENDER_EMAIL = "xingqiao.sjtu@gmail.com"
SENDER_PASSWORD = os.environ.get('EMAIL_PASSWORD')
API_KEY = os.environ.get('API_KEY')



class EmailService:

    def __init__(self, server, port):
        self.server_address = server
        self.port = port

    def send_email(self, receiver_email, subject, body):
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(self.server_address, self.port)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()

            return {"message": "Email has been sent successfully"}
        except Exception as e:
            return {"error": str(e)}


class GptService:

    def __init__(self):
        self.api_key = API_KEY
        openai.api_key = self.api_key

    def generate_event_info(self, tags: dict):
        requests = f"""generate campus event description using the info provided. If provided info is empty, exclude giving any infor about it.
keep the tone - {tags['tone']},
                        keep it around 100 words.
                                title is {tags['title']}, 
    `                           date is {tags['date']},
                                time is {tags['time']},
                                location is {tags['location']},
                                number of people is  {tags['number_of_people']},                  
                                other tags are {tags['tags']}
                                do not include anything with bracets[], make the text human-like
                                """

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": requests}]
        )

        answer = completion["choices"][0]["message"]["content"]
        return answer
