import sqlite3
import os
from dotenv import load_dotenv # to load env variables
import json
import openai
import datetime
from bs4 import BeautifulSoup
from util import get_filepath

load_dotenv()
openai.api_key  = os.getenv('OPENAI_API_KEY')

class SummarizeArticles:
    def __init__(self, db_filename):
        self.db_filename = db_filename
        self.conn = sqlite3.connect(self.db_filename)
        self.c = self.conn.cursor()
        # self.last_10_days_articles_count = 0
        self.trying_to_summarize = 0
        self.summarized_and_relevant = 0
        self.summary_attempts_limit_exceeded = 0
        self.summary_already_exists = 0
        self.summarized_but_not_relevant = 0
        self.token_limit_exceeded = 0
 
    def update_summarize_status(self, feed_id, article_id, number):
        self.c.execute('''
            UPDATE Metadata
            SET summarize_status = ?
            WHERE id = ? AND id_article = ?
            ''', (number, feed_id, article_id))
        self.conn.commit()

    def update_summarized_date(self, feed_id, article_id):
        today_date = str(datetime.datetime.today().strftime('%Y-%m-%d'))
        # print(f"Today's date: {today_date}")
        self.c.execute('''
            UPDATE Metadata
            SET summarized_date = ?
            WHERE id = ? AND id_article = ? AND summarize_status = ?
            ''', (str(today_date), feed_id, article_id, 1))
        self.conn.commit()

        return today_date

    def api_call(self, content):
        # API call to summarize the content
        MODEL = "gpt-3.5-turbo-1106"
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                            {
                                "role": "system", 
                                "content": """
                                Generate a concise, entity-dense summary of the given article, following these steps:
                                1. Read the article and identify the most important entities.
                                2. Write a dense summary, keeping it approximately 200 words long.

                                Guidelines & Characteristics of the summary:
                                - Relevant and true to the main article and it's content.
                                - Make sure your summaries are self-contained and clear without needing the article.
                                - Ensure the summary does not contain any irrelevant content, information or characters.

                                Ensure you handle the following special situations you may encounter while summarizing:
                                
                                Situation 1:
                                - Some articles only preview a few sentences and the rest will be hidden behind a paywall.
                                - Such cases would be abvious as the article would be incomplete.
                                - In those cases, Your response should exactly be "incomplete article unable to summarize" with no additional words or punctutations.

                                Situation 2:
                                - Sometimes some articles would be status updates on some logistics or operations regarding the substack of the writer
                                - In those cases, Your response should exactly be "not relevant unable to summarize" with no additional words or punctutations.
                                
                                Situation 3:
                                - If for any reason you are not able to generate a summary, Your response should exactly be "unable to summarize" with no additional words or punctutations.
                                """
                            },
                            {"role": "user", "content": f"This is the article that you have to summarize: {content}"}
                        ],
                temperature=0,
                request_timeout=30
            )

            summary = response["choices"][0]["message"]["content"]
            # print(f"Summary: \n{summary}\n")
            number = 1
            if "unable to summarize" in summary.lower():
                number = 2
        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                summary = "The article is too lengthy to be summarized."
                number = 3
            else:
                summary = "An error occurred while summarizing the article."
                number = 4
        except openai.error.Timeout as e:
            summary = "The API call timed out. Handle this as needed."
            number = 5
        return summary, number

    def add_summary_to_summaries_json(self, feed_id, article_id, title, guid, summary, published_date, summarized_date, file_path):
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
        else:
            data = []

        # Check if feed_id and title already exist
        existing_entry = next((item for item in data if item["feed_id"] == feed_id and item["title"] == title), None)
        
        # If the entry doesn't already exist, append the new data
        if not existing_entry:
            data.insert(0, {
                "feed_id": feed_id,
                "article_id": article_id,
                "title": title,
                "guid": guid,
                "published_date": published_date,
                "summarized_date": summarized_date,
                "summary": summary
            })

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def process_content(self, feed_id, id_article, title,  published_date):

        # Read the JSON file
        feed_folder = os.path.join('dbs/raw_feeds', str(feed_id))
        feed_json_path = os.path.join(feed_folder, 'feed.json')
        with open(feed_json_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        article_entry = None
        # Find entry (article) in the JSON file
        for entry in json_data:
            # print(f"Entry: {entry}")
            if entry.get('id_article') == id_article or entry.get('title') == title:
                article_entry = entry
                print(f"Found article for id: {feed_id}, id_article: {id_article}")
                break
        
        # If article entry exists
        if article_entry is not None:

            # Try to parse before api call to reduce token length, if cannot, send unparsed
            try:
                soup = BeautifulSoup(article_entry['content'][0]['value'], 'html.parser')
                parsed_content = soup.get_text()
                content = parsed_content
            except:
                content = article_entry['content'][0]['value']

            # Call the API to summarize the content
            summary, summary_status = self.api_call(content)
            # print(f"Summary: \n{summary}\n------------------\n")
            # print(f"Summary status: {number}\n------------------\n")

            # Update the Metadata table with the summary details
            self.update_summarize_status(feed_id, id_article, summary_status)
            summarized_date = self.update_summarized_date(feed_id, id_article)
            self.c.execute('''
                UPDATE Metadata
                SET summary_attempts = summary_attempts + 1
                WHERE id = ? AND id_article = ?
            ''', (feed_id, id_article))
            self.conn.commit()

            # Update the feed JSON with the summary. We don't need to update the feed json but we'll do it anyway.
            article_entry['summary'] = summary
            article_entry['published'] = published_date

            if summary_status == 1: # only when the summarization is successful
                self.summarized_and_relevant += 1
                if not os.path.exists('docs'):
                    os.mkdir('docs')
                summary_path = os.path.join('docs', 'summaries.json')
                self.add_summary_to_summaries_json(feed_id, id_article, article_entry['title'], article_entry['guid'], summary, published_date, summarized_date, summary_path)

            with open(feed_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)

        else:
            print(f"No matching article found for id: {feed_id}, id_article: {id_article}")

    def check_if_content_exists(self):
        # For each unique feed
        self.c.execute('SELECT DISTINCT id FROM LINKS')
        feed_ids = self.c.fetchall()

        for feed_id in feed_ids[:]:
            feed_id = feed_id[0]  # Extract the integer value from the tuple
            # print(f"feed_id: {feed_id}")
            # Select all articles for that specific feed
            self.c.execute('''
                SELECT id_article, title, summarize_status, summary_attempts, published_date
                FROM Metadata
                WHERE id = ?
            ''', (feed_id,))
            rows = self.c.fetchall()
            if len(rows) > 0:
                for row in rows[:]:
                    
                    id_article, title, summarize_status, summary_attempts, published_date = row
                    # print(f"Feed ID: {feed_id}: Article: {id_article}: Row length: {len(rows)}")
                    # If summary doesn't exist, or failed previously (status 4 or 5)
                    if summarize_status == 0 or summarize_status == 4 or summarize_status == 5:
                        self.trying_to_summarize += 1
                        # If summary_attempts is less than 2
                        if summary_attempts <= 3:
                            print(f"Parsing content for id: {feed_id}, id_article: {id_article}")
                            self.process_content(feed_id, id_article, title, published_date)
                            continue
                        else:
                            self.summary_attempts_limit_exceeded += 1
                            print(f"Summary attempts exceeded for id: {feed_id}, id_article: {id_article}, skipping...")
                            continue
                    elif summarize_status == 1:
                        self.summary_already_exists += 1
                        # print(f"Summary exists for id: {feed_id}, id_article: {id_article}, skipping...")
                        continue
                    elif summarize_status == 2:
                        self.summarized_but_not_relevant += 1
                        print(f"Summary is not relevant for id: {feed_id}, id_article: {id_article}, skipping...")
                        continue
                    elif summarize_status == 3:
                        self.token_limit_exceeded += 1
                        print(f"Summary is too long for id: {feed_id}, id_article: {id_article}, skipping...")
                        continue
        # self.conn.close()

if __name__ == '__main__':
    today_date = datetime.datetime.today().strftime('%Y-%m-%d')
    json_status_file_name = get_filepath('status.json')

    try:
        with open(json_status_file_name, 'r') as f:
            existing_status = json.load(f)
        
        # default status
        status_data = {
                'status': 'Did not run',
                'message': 'if elif conditions not met, check code'
            }
            
        if today_date in existing_status and 'GetArticles' in existing_status[today_date] and existing_status[today_date]['GetArticles']['status'] == 'Failed':
            status_data = {
                'status': 'Failed',
                'message': 'GetArticles failed so this has not run'
            }
        elif today_date in existing_status and 'GetArticles' in existing_status[today_date] and existing_status[today_date]['GetArticles']['status'] == 'Success':
            try:
                db_path = get_filepath('dbs/rss_sum.db')
                summarize_articles = SummarizeArticles(db_path)
                summarize_articles.check_if_content_exists()
                summarize_articles.conn.close()
                status_data = {
                    'status': 'Success',
                    'trying_to_summarize': summarize_articles.trying_to_summarize,
                    'summarized_and_relevant': summarize_articles.summarized_and_relevant,
                    'summary_already_exists': summarize_articles.summary_already_exists,
                    'summarized_but_not_relevant': summarize_articles.summarized_but_not_relevant,
                    'token_limit_exceeded': summarize_articles.token_limit_exceeded,
                    'summary_attempts_limit_exceeded': summarize_articles.summary_attempts_limit_exceeded
                }
            except Exception as e:
                status_data = {
                    'status': 'Failed',
                    'message': str(e)
                }

        existing_status[today_date]['SummarizeArticles'] = status_data

        with open(json_status_file_name, 'w') as f:
            json.dump(existing_status, f, indent=4)
    
    except FileNotFoundError:
        print(f"FileNotFoundError: {json_status_file_name} not found.")