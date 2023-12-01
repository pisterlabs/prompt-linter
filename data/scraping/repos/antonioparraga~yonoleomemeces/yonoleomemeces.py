import schedule
import time
import imaplib
import email
import html2text
import os
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import email
import smtplib


openai.api_key = os.getenv("OPENAI_API_KEY")
username = os.getenv("EMAIL_USERNAME")
password = os.getenv("EMAIL_PASSWORD")
email_address_to_send_to = "your_email_here@example.com"
filter_emails = False


#read imap from hotmail
def read_imap():
    print("Reading IMAP ...")
    try:
        mail = imaplib.IMAP4_SSL('imap-mail.outlook.com')
        mail.login(username, password)
        mail.list()
        mail.select('inbox')
        result, data = mail.uid('search', None, "UNSEEN")
        i = len(data[0].split())
        for x in range(i):
            latest_email_uid = data[0].split()[x]
            result, email_data = mail.uid('fetch', latest_email_uid, '(RFC822)')
            raw_email = email_data[0][1]
            #mark as read before anything else
            mail.uid('store', latest_email_uid, '+FLAGS', '(\Seen)')
            raw_email_string = raw_email.decode('utf-8')
            email_message = email.message_from_string(raw_email_string)
            email_from = email_message['From']
            if need_to_summarize_email_origin(email_from):
                subject = email_message['Subject']
                plain_text = ""
                html_text = ""
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True)
                        plain_text = plain_text + body.decode('utf-8')
                    elif part.get_content_type() == "text/html":
                        html_body = part.get_payload()
                        h = html2text.HTML2Text()
                        h.ignore_links = True
                        h.ignore_images = True
                        h.ignore_emphasis = True
                        h.ignore_tables = True
                        h.nobs = True
                        h.utf8 = True
                        html_text = html_text + h.handle(html_body)

                if plain_text != "": #prefer text version
                    text = plain_text
                else:
                    text = html_text

                if len(text) > 0:
                    #now summarize the text
                    print(text)
                    print("Summarizing...")
                    print("===========================================")
                    summary = summarize(text)
                    print(summary)
                    send_email(subject, summary)

    except Exception as e:
        print(str(e))

def need_to_summarize_email_origin(email_from):

    return_value = False #by default
    if filter_emails:
        email_origins = ["abogado@elquetengoaquicolgado.com",
                         "colegio@lespaganporpalabras.net"]

        for email in email_origins:
            if email in email_from:
                return_value = True
    else:
        return_value = True

    return return_value

def summarize(text):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=text + "\n\nResumen de 50 palabras:\n\n",
        temperature=0.7,
        max_tokens=200,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
        )
    return response.choices[0].text    

def send_email(subject, body):

    to_address = email_address_to_send_to

    msg = email.message_from_string(body)
    msg['From'] = username
    msg['To'] = to_address
    msg['Subject'] = "Resumen: " + subject

    server = smtplib.SMTP("smtp-mail.outlook.com", port=587)
    server.ehlo() # Hostname to send for this command defaults to the fully qualified domain name of the local host.
    server.starttls() #Puts connection to SMTP server in TLS mode
    server.ehlo()
    server.login(username, password)
    server.sendmail(username, to_address, msg.as_string().encode('utf-8'))
    server.quit()
    print("Email sent!")


#do every 1 minute
schedule.every(1).minutes.do(read_imap)

while 1:
    schedule.run_pending()
    time.sleep(1)
