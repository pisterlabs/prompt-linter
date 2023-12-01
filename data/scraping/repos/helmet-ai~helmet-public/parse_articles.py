import re
from html import unescape
from typing import List

import openai
from story import Article


def clean_content(content):
    # Remove HTML tags
    clean_html = re.sub('<.*?>', '', content)

    # Convert HTML entities to characters
    clean_text = unescape(clean_html)

    # Remove special characters and extra whitespace
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)

    return clean_text.strip()