# Standard library imports
import asyncio
import logging
import os
from threading import Lock

# Third-party imports
import aiohttp
import openai
from dotenv import load_dotenv
from msal import ConfidentialClientApplication
from transformers import GPT4Model, GPT4Tokenizer

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(filename='outlook_draft.log', level=logging.INFO)

# Initialize a lock for thread-safety
lock = Lock()

class EmailProcessor:
    def __init__(self, msal_client, openai_api, gpt4_model, gpt4_tokenizer):
        """
        Initializes an EmailProcessor object.

        Args:
            msal_client (ConfidentialClientApplication): The MSAL client for authentication.
            openai_api (str): The OpenAI API key.
            gpt4_model (GPT4Model): The GPT-4 model for generating draft replies.
            gpt4_tokenizer (GPT4Tokenizer): The GPT-4 tokenizer for encoding and decoding text.
        """
        self.msal_client = msal_client
        self.openai_api = openai_api
        self.gpt4_model = gpt4_model
        self.gpt4_tokenizer = gpt4_tokenizer

    async def get_unread_emails(self, session):
        """
        Retrieves unread emails from the Outlook API.

        Args:
            session (aiohttp.ClientSession): The aiohttp client session.

        Returns:
            dict: The JSON response containing the unread emails.
        """
        try:
            async with session.get('https://outlook.office.com/api/v2.0/me/mailfolders/inbox/messages?$filter=IsRead eq false') as resp:
                return await resp.json()
        except Exception as e:
            logging.error(f"Failed to fetch unread emails due to {str(e)}")
            return None

    async def generate_draft_reply(self, session, email):
        """
        Generates a draft reply for an email using the GPT-4 model.

        Args:
            session (aiohttp.ClientSession): The aiohttp client session.
            email (dict): The email to generate a draft reply for.

        Returns:
            str: The generated draft reply.
        """
        try:
            inputs = self.gpt4_tokenizer.encode(email['Body']['Content'], return_tensors='pt')
            outputs = self.gpt4_model.generate(inputs, max_length=500, num_return_sequences=1)
            reply = self.gpt4_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return reply
        except Exception as e:
            logging.error(f"Failed to generate draft reply due to {str(e)}")
            return None

    async def save_draft_reply(self, session, draft_content, email):
        """
        Saves a draft reply for an email in the Outlook API.

        Args:
            session (aiohttp.ClientSession): The aiohttp client session.
            draft_content (str): The content of the draft reply.
            email (dict): The email to reply to.
        """
        try:
            draft = {
                'Subject': f"RE: {email['Subject']}",
                'Body': {
                    'ContentType': 'Text',
                    'Content': draft_content
                },
                'ToRecipients': [
                    {
                        'EmailAddress': {
                            'Address': email['From']['EmailAddress']['Address']
                        }
                    }
                ]
            }
            await session.post('https://outlook.office.com/api/v2.0/me/messages', json=draft)
        except Exception as e:
            logging.error(f"Failed to save draft reply due to {str(e)}")

    async def process_email(self, session, email):
        """
        Processes an email by generating a draft reply and saving it as a draft.

        Args:
            session (aiohttp.ClientSession): The aiohttp client session.
            email (dict): The email to process.
        """
        try:
            draft_content = await self.generate_draft_reply(session, email)
            if draft_content is not None:
                await self.save_draft_reply(session, draft_content, email)
        except Exception as e:
            logging.error(f"Failed to process email due to {str(e)}")

    async def main(self):
        """
        Main method for processing unread emails.

        This method retrieves unread emails, processes each email, and saves draft replies.
        """
        try:
            async with aiohttp.ClientSession() as session:
                emails = await self.get_unread_emails(session)
                if emails is not None:
                    tasks = [self.process_email(session, email) for email in emails]
                    await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"Failed to process emails due to {str(e)}")

if __name__ == "__main__":
    client_id = os.getenv('CLIENT_ID')  # Use os.getenv
    client_secret = os.getenv('CLIENT_SECRET')  # Use os.getenv
    authority = os.getenv('AUTHORITY')  # Use os.getenv
    msal_client = ConfidentialClientApplication(client_id, client_secret=client_secret, authority=authority)

    openai_api_key = os.getenv('OPENAI_API_KEY')  # Use os.getenv
    openai.api_key = openai_api_key

    gpt4_model = GPT4Model.from_pretrained('gpt-4')
    gpt4_tokenizer = GPT4Tokenizer.from_pretrained('gpt-4')

    processor = EmailProcessor(msal_client, openai_api_key, gpt4_model, gpt4_tokenizer)
    asyncio.run(processor.main())
