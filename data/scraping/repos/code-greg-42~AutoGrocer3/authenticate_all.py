import os
import boto3
import openai
from google.cloud import vision
from utils.print_messages import cook_tim_sys_prompt, cook_tim_intro

# NECESSARY ENV VARIABLES:
# 1. AWS_ACCESS_KEY_ID
# 2. AWS_SECRET_ACCESS_KEY
# 3. GOOGLE_APPLICATION_CREDENTIALS
# 4. OPENAI_API_KEY

# OPTIONAL ENV VARIABLES:
# 1. TWILIO_ACCOUNT_SID
# 2. TWILIO_AUTH_TOKEN

def auth_all(user_name, use_twilio=False):
    
    # aws auth
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not aws_key or not aws_secret_key:
        raise ValueError("AWS environment variables not set up correctly.")
    session = boto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret_key,
    )
    aws_s3_client = session.client('s3')
    aws_bucket_name = f"autogrocer-user-{user_name.strip().lower().replace(' ', '-')}"
    
    # Check if the bucket exists
    try:
        aws_s3_client.head_bucket(Bucket=aws_bucket_name)
    except:
        # If the bucket does not exist, create it
        aws_s3_client.create_bucket(Bucket=aws_bucket_name)
    try:
        aws_response = aws_s3_client.get_object(Bucket=aws_bucket_name, Key='chat_history.txt')
        chat_history = aws_response['Body'].read().decode('utf-8')
    except:
        # if chat history doesn't exist yet, init it
        chat_history = [{
            "role": "system",
            "content": cook_tim_sys_prompt
        },
        {
            "role": "assistant",
            "content": cook_tim_intro
        }]
    # print("AWS S3 initialized.")
    
    # google cloud auth
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError("Google Cloud environment variable not set up correctly.")
    google_cloud_client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)
    # print("Google Cloud client initialized.")
    
    # openai auth
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI environment variable not set up correctly.")
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    # print("OpenAI client initialized.")
    
    twilio_client = None
    if use_twilio:
        from twilio.rest import Client
        # twilio auth
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        if not account_sid or not auth_token:
            raise ValueError("Twilio environment variables not set up correctly.")
        twilio_client = Client(account_sid, auth_token)
        # print("Twilio client initialized.")
    
    return {
        "chat_history": chat_history, 
        "aws_s3_client": aws_s3_client,
        "aws_bucket_name": aws_bucket_name,
        "google_cloud_client": google_cloud_client,
        "twilio_client": twilio_client
    }