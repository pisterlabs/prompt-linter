import subprocess
import json
from openai import OpenAI
import time

api_key = "sk-JUBeexjClFqDZNmJNOiD5tSYqABQqL8yRqpjpz"  # Replace with your actual OpenAI API key
client = OpenAI(api_key=api_key)
last_processed_sms = {}

def process_new_messages():
    try:
        
        result = subprocess.run(['termux-sms-list', '-t', 'inbox'], capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            new_messages = json.loads(result.stdout)
            print("New SMS data captured successfully.")

            for key in new_messages:
                bodyIncludeText = "?" # Reply if sms ends with '?'
                body = key['body'].strip()
                bodyIncludeText = bodyIncludeText.strip()

                # Check if 'sender' is available in the dictionary, if not, use 'number'
                sender = key.get('sender', None)
                if sender is None:
                    sender = key.get('number', None)

                # If 'sender' and 'number' are both not available, skip to the next message
                if sender is None:
                    continue

                if sender not in last_processed_sms or key['received'] > last_processed_sms[sender]['received']:
                    if bodyIncludeText.lower() in body.lower() and key['read'] == 0:
                        print(key['body'])
                        prompt = f"USER\n{key['body']}\n"
                        completion = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": f"You are a funny assistant and I am your boss. Greet the sender: {sender} ,and in 80 words write a reply to the following text"},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.8,
                            max_tokens=100  # Adjust as needed
                        )
                        # Extract text from completion.choices[0].message
                        completion_text = completion.choices[0].message.content
                        # send sms via termux-sms-send
                        subprocess.run(["termux-sms-send", "-n", key.get('number', ''), completion_text])
                        
                        # Mark the SMS as read using termux-sms-inbox -r
                        subprocess.run(['termux-sms-inbox', '-r', str(key['_id'])])

                        # Update the last processed SMS for the sender
                        last_processed_sms[sender] = key
        else:
            print("Error: Unable to capture new SMS data.")
    except Exception as e:
        print("An error occurred: {}".format(e))

# Run the loop to listen for new messages
while True:
    process_new_messages()
    # Sleep for 2 minutes before checking again
    time.sleep(120)
