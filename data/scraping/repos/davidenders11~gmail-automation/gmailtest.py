import sys
import argparse
import logging
from gmailwrapper import Gmail
from openaiwrapper import OpenAI
from googleapiclient.discovery import build
import pandas as pd
import json

logging.basicConfig(level=logging.ERROR, format="%(levelname)s:%(asctime)s:%(message)s")
logger = logging.getLogger(__name__)

gmail = Gmail(logger)

query = f"from:andrew@xostrucks.com"

# last_thread_id = gmail.get_most_recent_message_ids(query)["threadId"]
ids = gmail.get_most_recent_message_ids(query)
id = ids["id"]
threadId = ids["threadId"]
# print(id, threadId)
references, reply, subject = gmail.get_message_headers(id)
print(references, reply, subject)

# Find the "References" and "In-Reply-To" headers and extract their "value" fields

# print(json.dumps(message, indent=2))
# print(last_thread_id)
# thread = gmail.get_thread(last_thread_id)
# # logger.info("Retrieved last thread with target recipients")

# # generate the draft and create it on gmail, this should draft a reply
# gmail.draft("content", "andrew@xostrucks.com", thread_id=last_thread_id)
