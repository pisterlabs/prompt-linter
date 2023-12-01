import imaplib
import email
from email.header import decode_header
from datetime import datetime, timedelta
import openai
import requests
import json

username = "" # your email address
password = "" # your email password

OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

mail = imaplib.IMAP4_SSL("outlook.office365.com") # can be changed to other platforms

one_week_ago = (datetime.now() - timedelta(days=1)).strftime("%d-%b-%Y")

mail.login(username, password)

mail.select("inbox")

status, messages = mail.search(None, f'(SINCE "{one_week_ago}")')
email_ids = messages[0].split()

bodies = [" "] * 5
bodyIndex = 0

print("beginning to parse emails")

print(len(email_ids))

for email_id in email_ids:
    _, msg = mail.fetch(email_id, "(RFC822)")
    for response_part in msg:
        if isinstance(response_part, tuple):
            email_message = email.message_from_bytes(response_part[1])

            from_ = email.utils.parseaddr(email_message["From"])[1]
            to = email.utils.parseaddr(email_message["To"])[1]

            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True)
                        if len(bodies[bodyIndex]) > 12000:
                            bodyIndex+=1
                        bodies[bodyIndex] += body.decode('utf-8')
            else:
                body = email_message.get_payload(decode=True)
                if len(bodies[bodyIndex]) > 12000:
                            bodyIndex+=1
                bodies[bodyIndex] += body.decode('utf-8')

for i in range(len(bodies)):
    bodies[i] = bodies[i].replace("\r", "")
    bodies[i] = bodies[i].replace("\n", "")
    bodies[i] = bodies[i].replace("\u200c", "")
    
    
ret = ""

question = "Below, i have provided the content of several emails i have received. i want to send myself a recap email that highlights importgant information and action items but that is not overwhelming. please cateogrize these emails into easily ingestible ways and draft me a recap email. Please include a maximum of 10 bullet points"

# New question
question = "I have provided the content of several emails I received this past week below. I want to send myself a recap email that highlights important information and action items but that is not overwhelming. Can you please categorize these emails (max of 5 distinct categories) into easily ingestible content with 5 bullet points and draft me a recap email of this week’s emails? This is very important: Only return the draft email, nothing else. Finally sign the email with Best, C@tchUp"

prompt = question + " : " + bodies[0]



print("starting call to open ai")


# old call
# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[{"role": "user", "content": prompt}]
# )

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": prompt}],)


ret = completion["choices"][0]["message"]["content"]

print('finished call to open ai')

print('starting email send')
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import base64

sender_email = username
sender_password = password

recipient_email = username

image_path = "CatchUp_logo.jpeg" 

with open(image_path, "rb") as image_file:
    image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode()

message = MIMEMultipart()
message['From'] = sender_email
message['To'] = recipient_email
message['Subject'] = "Your Weekly C@tchUP is Here!"

message.attach(MIMEText(ret, 'plain'))

server = smtplib.SMTP('smtp.office365.com', 587) # can be changed to other platforms
server.starttls()
server.login(sender_email, sender_password)

server.sendmail(sender_email, recipient_email, message.as_string())

server.quit()

print('finished sending email')

