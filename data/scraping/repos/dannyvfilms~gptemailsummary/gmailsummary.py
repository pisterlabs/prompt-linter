# Install required components
from flask import Flask, jsonify, request, render_template
import openai
import os
import sys
from io import StringIO
import pickle
import logging
import requests
import base64
import configparser
import re
import logging
import subprocess
import textwrap
import threading
import google.auth.transport.requests as tr_requests
import imaplib
import email
from email.header import decode_header
from email.charset import Charset, QP
import asyncio
from hypercorn.asyncio import serve
from hypercorn.config import Config
from collections import defaultdict

app = Flask(__name__)

# OpenAI API parameters
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
CUSTOM_PROMPT = os.environ.get('CUSTOM_PROMPT')
OPENAI_ENGINE = os.environ.get('OPENAI_ENGINE')
OPENAI_MAX_TOKENS = int(os.environ.get('OPENAI_MAX_TOKENS'))
OPENAI_TEMPERATURE = float(os.environ.get('OPENAI_TEMPERATURE'))

# IMAP Email parameters
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_PROVIDER = os.environ.get("EMAIL_PROVIDER")
EMAIL_PROVIDERS = {
    "gmail": "imap.gmail.com",
    "outlook": "imap-mail.outlook.com",
    "yahoo": "imap.mail.yahoo.com",
    "aol": "imap.aol.com",
    "icloud": "imap.mail.me.com",
    "zoho": "imap.zoho.com",
    "gmx": "imap.gmx.com",
    "fastmail": "imap.fastmail.com",
    "protonmail": "imap.protonmail.com",  # ProtonMail requires a paid plan and Bridge for IMAP access
    "office365": "outlook.office365.com",
    "mailru": "imap.mail.ru",
    "yandex": "imap.yandex.com",
    "cpanel": "mail.yourdomain.com",  # Replace 'yourdomain.com' with your actual domain
    "dovecot": "mail.yourdomain.com",  # Replace 'yourdomain.com' with your actual domain
    "courier": "mail.yourdomain.com",  # Replace 'yourdomain.com' with your actual domain
    "hmailserver": "mail.yourdomain.com",  # Replace 'yourdomain.com' with your actual domain
}

accounts = [
    {
        "email": "account1@gmail.com",
        "password": "redacted",
        "provider": "gmail"
    },
    {
        "email": "account2@outlook.com",
        "password": "redacted",
        "provider": "outlook"
    }
]

# Set up OpenAI API credentials
openai.api_key = OPENAI_API_KEY

# List that will store the latest fetched emails
latest_emails = []

# Variable that will store the total number of characters for emails
total_characters = 0
max_characters = int(os.environ.get('EMAIL_MAXCHARACTERS'))

# Variable to allow summary function to interrupt payload generation
email_fetch_lock = Lock()

# Create nice text boxes in logs
def format_text_with_boxes(text, design='shell', padding='a1l2'):
    echo_process = subprocess.Popen(['echo', text], stdout=subprocess.PIPE)
    boxes_process = subprocess.Popen(['boxes', '-d', design, '-p', padding], stdin=echo_process.stdout, stdout=subprocess.PIPE, text=True)

    echo_process.stdout.close()  # Allow echo_process to receive a SIGPIPE if boxes_process exits.
    output = boxes_process.communicate()[0]
    echo_process.wait()
    return output

# Add the count_characters function
def count_characters(text):
    global total_characters
    char_count = len(text)
    total_characters += char_count
    print("Input:\n", text, flush=True)
    
    # Print the current character count
    count_message = format_text_with_boxes(f"Character count for this email: {char_count}\nTotal characters processed so far: {total_characters}")
    print(count_message, flush=True)

async def fetch_latest_emails(email_event):        
    global latest_emails, total_characters

    # Initialize an empty list to store the latest emails
    latest_emails = []

    # Initialize a variable to store the total character count
    total_characters = 0

    # Read environment variables
    variable_quantity = os.environ.get("EMAIL_VARIABLEQUANTITY", "false").lower() == "true"
    max_emails = int(os.environ.get("EMAIL_MAXEMAILS", 50))
    max_characters = int(os.environ.get("EMAIL_MAXCHARACTERS", 10000))  # Added EMAIL_MAXCHARACTERS variable

    # Print the while loop settings
    start_message = format_text_with_boxes(f"Creating Email Payload. Settings for the run:\n \nVariable Quantity: {variable_quantity}\nMax characters: {max_characters}\nMax emails: {max_emails}\n \nStarting the loop.")
    print(start_message, flush=True)

    imaps = []
    message_numbers_list = []
    processed_emails_ids_list = []

    # Iterate over the accounts and fetch emails
    for account in accounts:
        email_address = account["email"]
        email_password = account["password"]
        email_provider = account["provider"]

        # Connect to the IMAP server
        imap_url = EMAIL_PROVIDERS.get(email_provider.lower(), "imap.gmail.com")
        imap = imaplib.IMAP4_SSL(imap_url)
        imap.login(email_address, email_password)

        # Select the mailbox
        imap.select("inbox")

        summarized_folder = create_and_return_summarized_folder(imap)

        # Search for all unread emails that are not spam and not in the Summarized folder
        status, message_numbers = imap.search('UTF-8', f'UNSEEN NOT X-GM-LABELS SPAM NOT X-GM-LABELS "{summarized_folder}"')
        message_numbers = message_numbers[0].split()

        # Reverse the message numbers to fetch newest to oldest
        message_numbers = message_numbers[::-1]

        # Add imap connection and message numbers to their respective lists
        imaps.append(imap)
        message_numbers_list.append(message_numbers)

        print(f"Initial message_numbers: {message_numbers}", flush=True)  # Debug statement added

    # Initialize the processed_email_ids_list
    processed_email_ids_list = [set() for _ in range(len(accounts))]

    # Add a flag to break out of the outer while loop
    stop_loop = False

    current_account_index = 0
    num_accounts = len(accounts)

    while not email_event.is_set() and not stop_loop:
        all_accounts_empty = True

        for account_index, account in enumerate(accounts):
            # Add a lock check at the beginning of the loop
            if email_fetch_lock.locked():
                print("Email fetch lock acquired by another task. Stopping this iteration.")
                break
            
            email_address = account["email"]
            email_password = account["password"]
            email_provider = account["provider"]

            # Get the current imap connection and message numbers
            imap = imaps[account_index]
            message_numbers = message_numbers_list[account_index]

            # Update the message_numbers list to include new unread, non-spam emails
            status, new_message_numbers = imap.search('UTF-8', 'UNSEEN')
            new_message_numbers = new_message_numbers[0].split()
            message_numbers = list(set(message_numbers) | set(new_message_numbers))

            # Update the message_numbers_list for the current account
            message_numbers_list[account_index] = message_numbers

            # Remove processed email IDs from message_numbers
            message_numbers = [msg_num for msg_num in message_numbers if msg_num not in processed_email_ids_list[account_index]]

            if not message_numbers:
                print(f"No new emails found for account {email_address}.", flush=True)
                continue

            all_accounts_empty = False
            message_index = 0

            if message_index < len(message_numbers):
                # Add a lock check at the beginning of the loop
                if email_fetch_lock.locked():
                    print("Email fetch lock acquired by another task. Stopping this iteration.")
                    break
            
                message_number = message_numbers[message_index]
                message_index += 1
                _, msg_data_response = imap.fetch(message_number, "(BODY.PEEK[])")

                for response_part in msg_data_response:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject, _ = decode_header(msg["Subject"])[0]
                        sender, _ = decode_header(msg["From"])[0]

                        if isinstance(subject, bytes):
                            subject = subject.decode()
                        if isinstance(sender, bytes):
                            sender = sender.decode()

                        decoded_body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == 'text/plain':
                                    charset = part.get_content_charset()  # Get the charset from the email part
                                    if charset:  # If charset is found, use it for decoding
                                        try:
                                            decoded_body = part.get_payload(decode=True).decode(charset, errors='replace')
                                        except LookupError:  # If the provided charset is not recognized, fall back to utf-8
                                            decoded_body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                                    else:  # If charset is not found, use utf-8
                                        decoded_body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                                    break
                        else:
                            decoded_body = msg.get_payload(decode=True).decode(errors='replace')

                        # Call remove_html_and_links on decoded_body to get the cleaned text
                        cleaned_body = remove_html_and_links(decoded_body)

                        # Construct the decoded_data string with the cleaned body
                        decoded_data = f"Sender: {sender}\nSubject: {subject}\nBody: {cleaned_body}"

                        # Count characters in the decoded_data
                        count_characters(decoded_data)

                        if variable_quantity and total_characters >= max_characters:
                            stop_loop = True
                            break
                        elif not variable_quantity and len(latest_emails) >= max_emails:
                            stop_loop = True
                            break

                        mail_data = {
                            'id': message_number.decode(),
                            'account': email_address,  # Add the account identifier
                            'subject': subject,
                            'from': sender,
                            'body': cleaned_body,
                            'internalDate': email.utils.parsedate_to_datetime(msg["Date"]).timestamp() * 1000
                        }

                        latest_emails.append(mail_data)

                # Remove the processed email from the message_numbers list
                message_numbers = message_numbers[1:]
                message_numbers_list[account_index] = message_numbers
                processed_email_ids_list[account_index].add(message_number)

        # Move to the next account in a round-robin fashion
        current_account_index = (current_account_index + 1) % num_accounts

        # Get the updated account information
        account = accounts[current_account_index]
        email_address = account["email"]
        email_password = account["password"]
        email_provider = account["provider"]

        # Update the imap connection and message numbers for the next account
        imap = imaps[current_account_index]
        message_numbers = message_numbers_list[current_account_index]

        if all_accounts_empty:
            print("No new emails found in all accounts. Exiting loop.", flush=True)
            break

    # Sort the emails by internal date (oldest first)
    sorted_emails = sorted(latest_emails, key=lambda email: email['internalDate'])

    # Update the latest_emails global variable
    latest_emails = sorted_emails

    # Find the emails with the given ids in the latest_emails list
    emails = [e for e in latest_emails if e.get('subject', '') != 'No subject' and e.get('body', '') != 'No content']

    # Get the number of emails summarized
    num_emails = len(emails)

    # Get the unique senders' names
    unique_senders = set([e['from'] for e in emails])
    unique_senders_str = ', '.join(unique_senders)

    # Wrap the unique senders' string to 150 characters
    unique_senders_str = textwrap.fill(unique_senders_str, width=150)

    # Print the latest_emails list to the terminal
    print("\n" + "#" * 45 + "\n    Latest Emails Fetched:   \n" + "#" * 45 + "\n", flush=True)
    print(f"{latest_emails}", flush=True)

    payload_ready = format_text_with_boxes(f"Payload Generated.\n \nAmount: {num_emails}\nSenders: {unique_senders_str}\n \nSettings for the run:\n \nVariable Quantity: {variable_quantity}\nMax characters: {max_characters}\nMax emails: {max_emails}")
    print(payload_ready, flush=True)

    return latest_emails

async def start_email_monitor():
    email_event = asyncio.Event()
    while True:
        await fetch_latest_emails(email_event)
        await asyncio.sleep(60)  # Adjust the interval as needed

def remove_html_and_links(text):
    # Convert empty strings
    if text is None:
        text = ""
    # Remove unwanted HTML code
    text = re.sub('<table.*?</table>', '', text, flags=re.DOTALL)
    
    # Remove HTML tags
    text = re.sub('<[^<]+?>', '', text)

    # Remove links
    text = re.sub(r'http\S+', '', text)

    # Remove CSS styles
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', '', text, flags=re.DOTALL)

    # Remove JavaScript code
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)

    # Remove comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove consecutive whitespace characters
    text = re.sub('\s+', ' ', text)

    # Remove font code and Unicode code
    text = re.sub(r'(@font-face.*?;})', '', text, flags=re.DOTALL)
    text = re.sub(r'unicode-range:.*?;', '', text)

    # Remove unwanted code
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Remove blocks of code like &nbsp;&zwnj;&nbsp;&zwnj;...
    text = re.sub(r'(&\w+;)*[\u200b\u200c]*[\u00a0\u200b]*', '', text)

    # Remove code blocks
    text = re.sub(r'{[^{}]*}', '', text, flags=re.DOTALL)

    # Remove a:visited and a:link
    text = re.sub(r'a:visited|a:link', '', text)

    # Remove patterns like a[class="btn"], div[class="column"], etc.
    text = re.sub(r'\w+\[class="[^"]+"\]', '', text)

    # Remove multiple consecutive comment tags
    text = re.sub(r'(<!--\s*)+', '', text)

    # Remove specific words followed by a potential comma
    text = re.sub(r'\b(p|span|font|td|div|li|blockquote|table|img|h1|h2|h3|h4|h5|ol|ul|th)(,)?\b', '', text)

    # Remove consecutive commas
    text = re.sub(r'(,\s*){2,}', ', ', text)

    # Remove single words that begin with a period, asterisk, hash, hyphen, colon, or at symbol
    matches = list(re.finditer(r'(\s|^)(\.|\*|#|-|:|@)[^\s]+', text))
    if matches:
        offset = 0
        for match in matches:
            start, end = match.span()
            text = text[:start - offset] + text[end - offset:]
            offset += end - start

    # Remove extra spaces and multiple consecutive newline characters
    text = re.sub('\s+', ' ', text)
    text = re.sub('\n{2,}', '\n', text)

    return text.strip()

def create_and_return_summarized_folder(imap):
    try:
        status, _ = imap.create("Summarized")
        if status == "OK":
            print("Summarized folder created.")
        elif status == "NO":
            print("Summarized folder already exists.")
    except Exception as e:
        print(f"An error occurred while creating Summarized folder: {e}")
    
    return "Summarized"

def mark_emails_read(email_ids):
    # Group email IDs by account_identifier
    grouped_email_ids = defaultdict(list)
    for email_id, account_identifier in email_ids:
        grouped_email_ids[account_identifier].append(email_id)

    # Process emails for each account
    for account_identifier, account_email_ids in grouped_email_ids.items():
        # Find the account associated with the account_identifier
        account = next(acc for acc in accounts if acc['email'] == account_identifier)

        email_address = account["email"]
        email_password = account["password"]
        email_provider = account["provider"]

        # Connect to the IMAP server
        imap_url = EMAIL_PROVIDERS.get(email_provider.lower(), "imap.gmail.com")
        imap = imaplib.IMAP4_SSL(imap_url)
        imap.login(email_address, email_password)

        # Select the mailbox
        imap.select("inbox")

        # Create the Summarized folder if it doesn't exist
        summarized_folder = create_and_return_summarized_folder(imap)

        # Process each email ID for the current account
        for email_id in account_email_ids:
            try:
                # Convert the email ID to a string and set the \Seen flag for the email
                email_id_str = str(email_id)
                imap.store(email_id_str, '+FLAGS', '\\Seen')
                print(f"Marked email {email_id} as read.")
            except Exception as e:
                print(f"An error occurred while marking email {email_id} as read: {e}")

            # Move the email to the Summarized folder
            try:
                imap.copy(email_id_str, summarized_folder)
                imap.store(email_id_str, '+FLAGS', '\\Deleted')
                print(f"Moved email {email_id} to the Summarized folder.")
            except Exception as e:
                print(f"An error occurred while moving email {email_id} to the Summarized folder: {e}")

        # Expunge the mailbox to commit changes immediately
        imap.expunge()

        # Close the mailbox and logout from the IMAP server
        imap.close()
        imap.logout()
    
def mark_emails_read_async(email_ids):
    try:
        mark_emails_read(email_ids)
        print("Successfully marked emails as read.", flush=True)
    except Exception as e:
        print(f"Error marking emails as read: {e}", flush=True)
    
@app.route('/mark_emails_read', methods=['POST'])
def mark_emails_read_route():
    email_ids_str = request.json['email_ids']
    
    # Print the incoming email_ids_str for debugging purposes
    print(f"Incoming email_ids_str: {email_ids_str}", flush=True)

    # Split the string by newlines and remove empty strings
    email_ids_lines = [line for line in email_ids_str.split('\n') if line]

    # Group the lines into pairs and convert them into tuples
    email_ids = [(email_ids_lines[i], email_ids_lines[i+1]) for i in range(0, len(email_ids_lines), 2)]

    # Start a separate thread to mark the emails as read
    threading.Thread(target=mark_emails_read_async, args=(email_ids,), daemon=True).start()

    return jsonify({"success": True})

@app.route('/get_emails_summary', methods=['POST'])
def get_emails_summary_route():
    # Call the get_emails_summary function with the latest_emails list
    return get_emails_summary(latest_emails)

def get_emails_summary(latest_emails):
    with email_fetch_lock:   
        error_message = ""
        num_emails = 0
        unique_senders_str = ""
        email_content = ""

        # Print the received email IDs
        print(f"Received email IDs: {[e['id'] for e in latest_emails]}", flush=True)

        try:
            # Find the emails with the given ids in the latest_emails list
            emails = [e for e in latest_emails if e.get('subject', '') != 'No subject' and e.get('body', '') != 'No content']

            # Get the number of emails summarized
            num_emails = len(emails)

            # Get the unique senders' names
            unique_senders = set([e['from'] for e in emails])
            unique_senders_str = ', '.join(unique_senders)

            # Wrap the unique senders' string to 150 characters
            unique_senders_str = textwrap.fill(unique_senders_str, width=150)

            # Concatenate the email content
            for i, email in enumerate(emails):
                email_content += f"Email {i + 1}:\nSubject: {email['subject']}\nFrom: {email['from']}\n\n{email['body']}\n\n"

        except Exception as e:
            error_message = f"An error occurred while processing the emails: {e}"

        if error_message:
            prompt = f"{CUSTOM_PROMPT}\n\n{error_message}"
        elif num_emails == 0:
            prompt = f"Assume you are a friendly assistant. You have just checked to see if there are any new noteworthy emails to summarize. No emails are available, so let the user know that there are no new emails to report and provide reassurance that you're keeping an eye on their inbox. Ensure that your response works best when spoken and maintain a tone that demonstrates emotional intelligence."
        else:
            prompt = f"{CUSTOM_PROMPT}\n\n{email_content}"

        # Print the prompt to the console
        print("\n" + "#" * 45 + "\n    Generated Prompt:   \n" + "#" * 45 + "\n", flush=True)
        print(prompt, flush=True)

        # Generate a summary of the emails using the OpenAI API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

        data = {
            "model": OPENAI_ENGINE,
            "messages": [
                {"role": "system", "content": prompt},
            ],
            "max_tokens": OPENAI_MAX_TOKENS,
            "n": 1,
            "temperature": OPENAI_TEMPERATURE,
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
            )
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to OpenAI API: {e}", flush=True)
            return jsonify({"error": "Failed to connect to OpenAI API."})
        except ValueError as e:
            print(f"Error parsing OpenAI API response: {e}", flush=True)
            return jsonify({"error": "Failed to parse OpenAI API response."})

        try:
            response_json = response.json()
        except Exception as e:
            print(f"Error: {e}", flush=True)
            return jsonify({"error": str(e)})

        print("\n" + "#" * 45 + "\n    OpenAI Response Payload:   \n" + "#" * 45 + "\n", flush=True)
        print(response_json, flush=True)

        if 'error' in response_json:
            print(f"OpenAI API error: {response_json['error']['message']}", flush=True)
            return jsonify({"error": response_json["error"]["message"]})

        try:
            summary = response_json["choices"][0]["message"]["content"].strip()
        except KeyError:
            print("Error extracting summary from OpenAI API response.", flush=True)
            return jsonify({"error": "Failed to extract summary from OpenAI API response."})

        # Extract token usage from response JSON
        usage = response_json['usage']
        prompt_tokens = usage['prompt_tokens']
        completion_tokens = usage['completion_tokens']
        total_tokens = usage['total_tokens']

        # Return the summary as plain text instead of JSON
        print("\n" + "#" * 45 + "\n    OpenAI Response:   \n" + "#" * 45 + "\n", flush=True)
        print(summary, flush=True)

        # Wrap the summary string to 150 characters per line
        summary_wrapped = textwrap.fill(summary, width=150)

        final_summary = format_text_with_boxes(f"Run Completed Successfully! \n \nAmount: {num_emails}\nSenders: {unique_senders_str}\n \nTokens Sent: {prompt_tokens}\nTokens Received: {completion_tokens}\nTotal Tokens: {total_tokens}\n \nResponse: \n{summary_wrapped}")
        print(final_summary)

        statistics = f"Summarized {num_emails} Emails. \n \nTokens Sent: {prompt_tokens}\nTokens Received: {completion_tokens}\nTotal Tokens: {total_tokens} \n \n Senders: {unique_senders_str}"

        # Start a separate thread to mark the emails as read
        email_ids = [(e['id'], e['account']) for e in emails]

        return jsonify({"summary": summary, "statistics": statistics, "email_ids": email_ids})

# WebUI
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(start_email_monitor())
    
    config = Config()
    config.bind = ['0.0.0.0:1337']  # Update the host and port as needed
    loop.run_until_complete(serve(app, config))
