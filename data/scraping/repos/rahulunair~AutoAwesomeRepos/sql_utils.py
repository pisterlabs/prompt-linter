import pandas as pd
import sqlite3
import os
from datetime import datetime
from openai_utils import get_relevancy_score_and_reasons
from openai_utils import classify_readme_category
from openai_utils import summarize_readme
from utils import classify_readme
from summary import generate_summary as summary
from utils import print

import re
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt")

from rich.progress import Progress

current_timestamp = datetime.utcnow()


def export_table_to_csv_pandas(filename):
    """Export repos db to csv."""
    conn = sqlite3.connect("./db/repos.sqlite")
    df = pd.read_sql_query("SELECT * FROM repo_details", conn)
    df.to_csv(filename, index=False)
    print(f"Data exported to {filename}")
    conn.close()

def print_top_k_pandas(k, order_by='stars_count', ascending=False):
    """Print top k records ordered by repos db."""
    conn = sqlite3.connect("./db/repos.sqlite")
    df = pd.read_sql_query("SELECT * FROM repo_details", conn)
    print(f"processing {len(df)} repos, please wait...")
    df = df.sort_values(by=order_by, ascending=ascending).head(k)
    print(f"Top {k} records ordered by {order_by}:")
    print(df)
    conn.close()

def create_database_table():
    """Create database table for repo data."""
    if not os.path.exists('./db'):
        os.makedirs('./db')
    conn = sqlite3.connect("./db/repos.sqlite")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS repo_details
                (id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                license TEXT DEFAULT 'Unknown License',
                readme TEXT DEFAULT '',
                stars_count INTEGER DEFAULT 0,
                forks_count INTEGER DEFAULT 0,
                pushed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                languages TEXT DEFAULT '',
                topics TEXT DEFAULT '',
                open_issues INTEGER DEFAULT 0,
                closed_issues INTEGER DEFAULT 0,
                description TEXT DEFAULT '',
                fork INTEGER DEFAULT 0,
                size INTEGER DEFAULT 0,
                watchers_count INTEGER DEFAULT 0,
                language TEXT DEFAULT '',
                keyword TEXT DEFAULT '',
                additional_keywords TEXT DEFAULT '',
                is_relevant INTEGER DEFAULT NULL,
                brief_desc TEXT DEFAULT NULL,
                class_label TEXT DEFAULT NULL,
                num_stars INTEGER DEFAULT 0,
                days_ago INTEGER DEFAULT 0,
                fetch_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def create_history_database_table():
    conn = sqlite3.connect("./db/repos.sqlite")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS repo_history (
            id INTEGER NOT NULL,
            url TEXT,
            license TEXT DEFAULT 'Unknown License',
            readme TEXT DEFAULT '',
            stars_count INTEGER DEFAULT 0,
            forks_count INTEGER DEFAULT 0,
            pushed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            languages TEXT DEFAULT '',
            topics TEXT DEFAULT '',
            open_issues INTEGER DEFAULT 0,
            closed_issues INTEGER DEFAULT 0,
            description TEXT DEFAULT '',
            fork INTEGER DEFAULT 0,
            size INTEGER DEFAULT 0,
            watchers_count INTEGER DEFAULT 0,
            language TEXT DEFAULT '',
            keyword TEXT DEFAULT '',
            additional_keywords TEXT DEFAULT '',
            is_relevant INTEGER DEFAULT NULL,
            brief_desc TEXT DEFAULT NULL,
            class_label TEXT DEFAULT NULL,
            num_stars INTEGER DEFAULT 0,
            days_ago INTEGER DEFAULT 0,
            fetch_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            timestamp TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def initialize_repo_history():
    """Initialize repo history table with existing repo details."""
    conn = sqlite3.connect("./db/repos.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM repo_details")
    repo_details_data = cursor.fetchall()
    for row in repo_details_data:
        cursor.execute("""
            INSERT INTO repo_history (
                id, url, license, readme, stars_count, forks_count, pushed_at,
                updated_at, created_at, languages, topics, open_issues,
                closed_issues, description, fork, size, watchers_count,
                language, keyword, additional_keywords, num_stars, days_ago, fetch_date, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (*row,datetime.utcnow(), datetime.utcnow()))
    conn.commit()
    conn.close()

def get_new_repos():
    """Get new repos from repo details table."""
    conn = sqlite3.connect("./db/repos.sqlite")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM repo_details
        WHERE id NOT IN (SELECT DISTINCT id FROM repo_history)
    """)
    new_repos = cursor.fetchall()
    conn.close()
    return new_repos

def reset_repo_info():
    """Reset the repo_details table."""
    conn = sqlite3.connect("./db/repos.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='repo_details'")
    table_exists = cursor.fetchone() is not None
    if table_exists:
        cursor.execute("DELETE FROM repo_details")
        conn.commit()
        print("Deleted all records from the 'repo_details' table.")
    else:
        print("Table 'repo_details' does not exist.")
    conn.close()

def save_results_to_db(repos, num_stars, days_ago):
    """Save repo data to database."""
    create_database_table()
    create_history_database_table()
    conn = sqlite3.connect("./db/repos.sqlite")
    cursor = conn.cursor()
    for repo_info in repos:
        cursor.execute("SELECT * FROM repo_details WHERE id=?", (repo_info.id,))
        existing_row = cursor.fetchone()
        if existing_row:
            existing_row_dict = dict(zip([description[0] for description in cursor.description], existing_row))
            existing_keywords = set(existing_row_dict['additional_keywords'].split('|'))
            if repo_info.keyword != existing_row_dict['keyword'] and repo_info.keyword not in existing_keywords:
                updated_additional_keywords = '|'.join(existing_keywords | {repo_info.keyword})
                cursor.execute(
                    "UPDATE repo_details SET additional_keywords=? WHERE id=?",
                    (updated_additional_keywords, repo_info.id)
                )
        else:
            cursor.execute("""
                INSERT INTO repo_details (
                    id, url, license, readme, stars_count, forks_count, pushed_at,
                    updated_at, created_at, languages, topics, open_issues,
                    closed_issues, description, fork, size, watchers_count,
                    language, keyword, additional_keywords, num_stars, days_ago, fetch_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                repo_info.id, repo_info.url, repo_info.license, repo_info.readme,
                repo_info.stars_count, repo_info.forks_count, repo_info.pushed_at,
                repo_info.updated_at, repo_info.created_at, repo_info.languages,
                '|'.join(repo_info.topics), repo_info.open_issues,
                repo_info.closed_issues, repo_info.description, repo_info.fork,
                repo_info.size, repo_info.watchers_count, repo_info.language,
                repo_info.keyword, '|'.join(repo_info.additional_keywords), num_stars, days_ago, current_timestamp,
            ))
        cursor.execute("""
            INSERT OR REPLACE INTO repo_history (
                id, url, license, readme, stars_count, forks_count, pushed_at,
                updated_at, created_at, languages, topics, open_issues,
                closed_issues, description, fork, size, watchers_count,
                language, keyword, additional_keywords, num_stars, days_ago, fetch_date, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            repo_info.id, repo_info.url, repo_info.license, repo_info.readme,
            repo_info.stars_count, repo_info.forks_count, repo_info.pushed_at,
            repo_info.updated_at, repo_info.created_at, repo_info.languages,
            '|'.join(repo_info.topics), repo_info.open_issues,
            repo_info.closed_issues, repo_info.description, repo_info.fork,
            repo_info.size, repo_info.watchers_count, repo_info.language,
            repo_info.keyword, '|'.join(repo_info.additional_keywords),
            num_stars, days_ago, current_timestamp, current_timestamp
        ))
    conn.commit()
    conn.close()  


def process_brief_desc(text):
    text = " ".join(text.split())
    sentences = sent_tokenize(text)
    return " ".join(sentences[:3])

def process_repo_details(force_update=False):
    """Process repo details and update database table."""
    conn = sqlite3.connect("./db/repos.sqlite")
    df = pd.read_sql_query("SELECT * FROM repo_details", conn)
    n = len(df)

    with Progress() as progress:
        task1 = progress.add_task("[cyan]Processing relevancy...", total=n)
        task2 = progress.add_task("[magenta]Processing summaries...", total=n)
        task3 = progress.add_task("[green]Processing class labels...", total=n)

        def update_progress(task):
            progress.update(task, advance=1)

        def get_relevancy_score(row):
            if pd.isnull(row['is_relevant']) or force_update:
                try:
                    result = get_relevancy_score_and_reasons(row['readme'])["score"]
                    update_progress(task1)
                except Exception:
                    result = "Error occurred, attempting to save processed relevancy scores..."
                return result
            return row['is_relevant']

        def get_summary(row):
            if pd.isnull(row['brief_desc']) or force_update:
                try:
                    result = summarize_readme(row['readme'])
                    update_progress(task2)
                except Exception:
                    result = "Error occurred, attempting to save processed summaries..."
                return result
            return row['brief_desc']

        def get_class_label(row):
            if pd.isnull(row['class_label']) or force_update:
                try:
                    result = classify_readme(readme=row['readme'], readme_summary=row['brief_desc'], use_openai=False)
                    update_progress(task3)
                except Exception:
                    result = "Error occurred,attempting to save processed class labels..."
                return result
            return row['class_label']

        df['is_relevant'] = df.apply(get_relevancy_score, axis=1)
        df['brief_desc'] = df.apply(get_summary, axis=1)
        df['class_label'] = df.apply(get_class_label, axis=1)

    df.to_sql('repo_details', conn, if_exists='replace', index=False)
    conn.close()


def process():
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"./results/repos_365_with_all_{timestamp_str}.csv"
    process_repo_details()   
    export_table_to_csv_pandas(fname)

if __name__ == "__main__":
    process()
    print_top_k_pandas(20, order_by='stars_count', ascending=False)
