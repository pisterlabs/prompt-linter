#!/usr/bin/env python3

# This file is part of Filter RSS Feed with GPT-4.
#
# Filter RSS Feed with GPT-4 is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Filter RSS Feed with GPT-4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Filter RSS Feed with GPT-4.  If not, see <https://www.gnu.org/licenses/>.

"""
Filter RSS feed based on GPT-4 score.

This script filters an RSS feed by requesting relevance scores from GPT-4 and
only including entries that meet a user-defined threshold.

Author: Nils Durner <ndurner+frss@googlemail.com>
"""

import argparse
import feedparser
import sqlite3
import openai
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict
from bs4 import BeautifulSoup


def html_to_plain_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Filter RSS feed based on GPT-4 score.")
    parser.add_argument("rss_feed_path", help="Path to the source RSS feed file.")
    parser.add_argument("rss_dest_path", help="Path to the destination RSS feed file.")
    parser.add_argument("system_prompt_path", help="Path to the text file containing the system prompt.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for filtering entries based on GPT-4 score (optional, default: 0.5).")
    args = parser.parse_args()

    # Get the OpenAI API key from the environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("Please set the 'OPENAI_API_KEY' environment variable")

    # Set the API key
    openai.api_key = openai_api_key

    # Connect to the SQLite database
    db = sqlite3.connect("gpt_scores.db")
    cursor = db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS scores (entry_id TEXT PRIMARY KEY, score REAL, timestamp TIMESTAMP)")

    # Parse the RSS feed
    try:
        with open(args.rss_feed_path, "rb") as rss_file:
            rss_feed = feedparser.parse(rss_file)
    except FileNotFoundError:
        print(f"Error: The input RSS feed file '{args.rss_feed_path}' was not found.")
        sys.exit(1)

    # Read the system prompt from the text file
    try:
        with open(args.system_prompt_path, "r") as sys_prompt_file:
            system_prompt = sys_prompt_file.read().strip()
    except FileNotFoundError:
        print(f"Error: The system prompt file '{args.system_prompt_path}' was not found.")
        sys.exit(1)

    # Iterate over the RSS feed entries and filter based on GPT-4 score
    filtered_entries = []
    for entry in rss_feed.entries:
        # Check if the entry score is already
        # cached in the SQLite database
        cursor.execute("SELECT score FROM scores WHERE entry_id = ?", (entry.id,))
        row = cursor.fetchone()

        score_retrieval_successful = False
        if row:
            print(f"got filter result cached for {entry.id}")
            score = row[0]
            score_retrieval_successful = True
        else:
            print(f"requesting filter result for {entry.id}")

            # If the score is not cached, ask GPT-4 for the score and cache it
            plain_description = html_to_plain_text(entry.description)
            prompt = f"{plain_description}"

            score_retrieval_successful = False

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=5,
                    temperature=0
                )
                score = float(response.choices[0].message['content'].strip())
                score_retrieval_successful = True
            except openai.OpenAIError as e:
                print(f"Error: An error occurred while making a request to the OpenAI API: {e}")
                score = 0.661
            except ValueError:
                score = 0.662
            
            if score_retrieval_successful:
                print(prompt, "\nscore:\n", score);
                cursor.execute("INSERT INTO scores (entry_id, score, timestamp) VALUES (?, ?, ?)", (entry.id, score, datetime.utcnow()))

        # If the score meets the threshold, add the entry to the filtered list
        if score >= args.threshold or not score_retrieval_successful:
            entry.score = score
            filtered_entries.append(entry)

    # Purge entries from the cache that are older than 10 days
    purge_timestamp = datetime.utcnow() - timedelta(days=10)
    cursor.execute("DELETE FROM scores WHERE timestamp < ?", (purge_timestamp,))
    db.commit()

    with open(args.rss_dest_path, "w") as outfile:
        outfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        outfile.write('<rss version="2.0">\n')
        outfile.write(f'  <channel>\n')
        outfile.write(f'    <title>{rss_feed.feed.title}</title>\n')
        outfile.write(f'    <link>{rss_feed.feed.link}</link>\n')
        outfile.write(f'    <description>{rss_feed.feed.description}</description>\n')

        for entry in filtered_entries:
            outfile.write(f'    <item>\n')
            outfile.write(f'      <title>{entry.title}</title>\n')
            outfile.write(f'      <link>{entry.link}</link>\n')
            outfile.write(f'      <description>{entry.description}\n\nscore: {entry.score}</description>\n')
            outfile.write(f'      <guid>{entry.id}</guid>\n')
            outfile.write('    </item>\n')

        outfile.write('  </channel>\n')
        outfile.write('</rss>\n')

    # Close the SQLite database connection
    db.close()


if __name__ == "__main__":
    main()
