import os
import json
import asyncio
from dotenv import load_dotenv
from email_fetcher import EmailFetcher
from tldr.tldr_splitter import TLDRSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import logging

from tldr.tldr_few_shot import system_message, example_in, example_out
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()  # Load environment variables from the .env file

class NewsletterProcessor:
    """
    NewsletterProcessor is a class to process Newsletter emails.
    It fetches emails, splits them into sections, and processes the content
    with a chat model.
    """
    def __init__(self, user_email: str, output_file: str, error_file: str, splitter_class, newsletter_email: str = 'dan@tldrnewsletter.com') -> None:
        self.user_email = user_email
        self.newsletter_email = newsletter_email
        self.email_fetcher = EmailFetcher(self.user_email)
        self.splitter = splitter_class()
        self.chat = ChatOpenAI(temperature=.4, max_retries=0, max_tokens=2400)
        self.output_file = output_file
        self.error_file = error_file
        self.messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=example_in),
            AIMessage(content=example_out)
        ]

    def fetch_emails(self) -> list[dict]:
        self.email_fetcher.connect()
        emails_d = self.email_fetcher.fetch_emails(self.newsletter_email, self.output_file)
        self.email_fetcher.disconnect()
        logging.info(f"Fetched {len(emails_d)} emails")
        return emails_d

    async def process_emails(self, emails: list[dict], concurrent_limit: int = 5) -> None:
        # Create the output and error files if they don't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as f:
                json.dump([], f, indent=4)
        if not os.path.exists(self.error_file):
            with open(self.error_file, 'w') as f:
                json.dump([], f, indent=4)

        # Create a semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrent_limit)

        # Process the emails
        tasks = []
        worker_num = 0
        for email_obj in emails:
            if 'body' in email_obj:
                parts = self.splitter.parse(email_obj['body'])
                for part in parts:
                    worker_num += 1
                    task = asyncio.ensure_future(
                        self._process_part(
                            part=part,
                            email_obj=email_obj,
                            sem=sem,
                            worker_num=worker_num
                        )
                    )
                    tasks.append(task)
        await asyncio.gather(*tasks)

    async def _process_part(self, part: str, email_obj: dict, sem: asyncio.Semaphore, worker_num: int) -> None:
        async with sem:
            email_subject = email_obj.get('subject', 'Unknown Subject')
            logging.info(f"[Worker {worker_num}] Processing part: {part[:50]}... (Email: {email_subject})")
            if not part:
                logging.info("Empty record. Skipping.")
                return
            to_run = self.messages.copy()
            to_run.append(HumanMessage(content=part))

            success = False
            retries = 3
            while not success and retries > 0:
                try:
                    chat_result = await self.chat._agenerate(to_run)
                    aimessage = chat_result.generations[0].message
                    obj = eval(aimessage.content)
                    success = True
                except Exception as e:
                    retries -= 1
                    logging.warning(f"[Worker {worker_num}] Processing failed, retries left: {retries}. Exception: {e}")

            if success:
                logging.info(f"[Worker {worker_num}] Processing successful for part: {part[:50]}... (Email: {email_subject})")
                for record in obj:
                    record['id'] = email_obj['id']
                    record['from'] = email_obj['from']
                    record['date'] = email_obj['date']
                    with open(self.output_file, 'r') as f:
                        l = json.load(f)
                        l.append(record)
                    with open(self.output_file, 'w') as f:
                        json.dump(l, f, indent=4)
            else:
                logging.error(f"[Worker {worker_num}] Processing failed after 3 retries for part: {part[:50]}... (Email: {email_subject})")
                error_data = {
                    'part': part,
                    'from': email_obj['from'],
                    'date': email_obj['date']
                }
                with open(self.error_file, 'r') as f:
                        l = json.load(f)
                        l.append(error_data)
                with open(self.error_file, 'w') as f:
                    json.dump(l, f, indent=4)

