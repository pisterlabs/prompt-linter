"""
Lambda to to pull URL from ETC channel messages
And post any URL to the newsletter generator
"""
import json
import os
import re
import random
import boto3
import requests
import validators
import openai

EXCLUDE_URL_STRINGS = ['skype.com', 'meet.google.com', 'trello.com/b']
QUEUE_URL = os.environ.get('SKYPE_SENDER_QUEUE_URL')


def clean_message(message):
    "Clean up the message received"
    message = message.replace("'", '-')
    message = message.replace('"', '-')

    return message

def get_message_contents(event):
    "Retrieve the message contents from the SQS event"
    record = event.get('Records')[0]
    message = record.get('body')
    message = json.loads(message)['Message']
    message = json.loads(message)

    return message

def process_reply(reply):
    "Strip out the title and summary"
    title = "FILL THIS PLEASE!"
    summary = "Dear Human, FILL THIS PLEASE!"

    if 'TITLE:' in reply and 'SUMMARY:' in reply:
        title = reply.split('TITLE:')[-1].split('SUMMARY:')[0].strip()
        summary = reply.split('SUMMARY:')[-1].strip()

    return title, summary

def ask_the_all_knowing_one(input_message, max_tokens=512):
    "Return the ChatGPT response"
    openai.api_key = os.environ.get('CHATGPT_API_KEY', '')
    model_engine = os.environ.get('CHATGPT_VERSION', 'gpt-3.5-turbo')

    input_message = "I want you to format your reply in a specific manner to this request." \
                    "I am going to send you an article (in quotes at the end of this message)." \
                    "You tell me its title and summary." \
                    "Use no more than 3 sentences for the summary." \
                    "Preface the title with the exact string TITLE: " \
                    "and preface the summary with the exact string SUMMARY:" \
                    "If you do not know, then put TITLE: UNKNOWN and SUMMARY: UNKNOWN." \
                    f"Ok, here is the article '{input_message}'"

    response = openai.ChatCompletion.create(
            model=model_engine,
            messages=[
                {"role": "user", "content": input_message},
            ],
            max_tokens=max_tokens
        )
    return response["choices"][0]["message"]["content"]

def get_title_summary(article_url):
    "Ask ChatGPT for the title and summary"
    reply = ask_the_all_knowing_one(article_url)

    return process_reply(reply)

def get_url(message):
    "Get the URL from the message"
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)"
    regex += r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))"
    regex += r"(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url_patterns = re.findall(regex,message)
    urls = []
    for url in url_patterns:
        if url[0][-1] != '-':
            present_flag = False
            for exclude_url in EXCLUDE_URL_STRINGS:
                if exclude_url in url[0]:
                    present_flag = True
                    break
            if not present_flag and validators.url(url[0]):
                urls.append(url[0])

    return urls

def post_to_newsletter(final_url, article_editor, category_id = '5'):
    "Method to call the newsletter API and post the url"
    url = os.environ.get('URL', '')
    category_id = os.environ.get('DEFAULT_CATEGORY', category_id)
    headers = {'x-api-key' : os.environ.get('API_KEY_VALUE','')}
    response_status = ""
    if len(final_url) != 0:
        for article_url in final_url:
            title, summary = get_title_summary(article_url)
            data = {'url': article_url,
                    'title': title,
                    'description': summary,
                    'category_id': category_id, 
                    'article_editor': article_editor}
            response = requests.post(url, data = data, headers = headers)
            response_status = response.status_code
            print(response_status)
    return response_status

def pick_random_user(article_editors_list):
    "Return a random employee to edit the article"
    tmp = article_editors_list[:]
    result = [tmp.pop(random.randrange(len(tmp))) for _ in range(1)]
    list_to_str = ' '.join(map(str, result))

    return list_to_str

def get_article_editor(employee_list):
    "Return a list of primary comment reviewers"
    return os.environ.get(employee_list,"").split(',')

def write_message(message, channel):
    "Send a message to Skype Sender"
    # Check if running on localstack or production environment
    is_localstack = os.environ.get('LOCALSTACK_ENV') == 'true'

    if is_localstack:
        sqs = boto3.client('sqs',endpoint_url= 'http://localstack:4566')
    else:
        sqs = boto3.client('sqs')
    print(channel)
    message = str({'msg':f'{message}', 'channel':channel})
    print(message)
    sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=(message))

def get_reply():
    "Get the Employee to edit the article for newsletter"
    article_editors_list = get_article_editor('employee_list')
    article_editor = pick_random_user(article_editors_list)
    reply = f'Article editor: {article_editor}'

    return reply,article_editor

def lambda_handler(event, context):
    """
    Method run when Lambda is triggered
    calls the filtering logic
    calls the logic to post to endpoint
    """
    content = get_message_contents(event)
    message = content['msg']
    channel = content['chat_id']
    user = content['user_id']
    print(f'{message}, {user}, {channel}')

    response=""
    final_url=[]
    if channel == os.environ.get('ETC_CHANNEL') and user != os.environ.get('Qxf2Bot_USER'):
        print("Getting message posted on ETC ")
        cleaned_message = clean_message(message)
        final_url=get_url(cleaned_message)
        #Filtered URL is printed by lambda
        print("Final url is :",final_url)
        if final_url:
            reply,article_editor = get_reply()
            response = post_to_newsletter(final_url, article_editor)
            write_message(reply, os.environ.get('ETC_CHANNEL',''))
        else:
            print("message does not contain any url")
    else:
        print("Message not from ETC channel")

    return {
        'statusCode': response,
        'body': json.dumps(final_url)
    }
