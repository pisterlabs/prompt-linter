from sqlalchemy import create_engine, text
import streamlit as st
import pandas as pd
import psycopg2
import uuid
import os

try:
    db_params = {
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASS"],
        "host": os.environ["DB_HOST"],
        "port": os.environ["DB_PORT"],
    }
except:
    db_params = {**st.secrets["postgres"]}


database_url = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"


def log_error_db(error):
    """Log error in DB along with streamlit app state."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        error_id = str(uuid.uuid4())
        tstp = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        query = text(
            """
            INSERT INTO error_logs (error_id, tstp, error)
            VALUES (:error_id, :tstp, :error);
        """
        )
        conn.execute(
            query,
            {
                "error_id": str(error_id),
                "tstp": tstp,
                "error": str(error),
            },
        )


def log_qna_db(user_question, response):
    """Log Q&A in DB along with streamlit app state."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        qna_id = str(uuid.uuid4())
        tstp = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        query = text(
            """
            INSERT INTO qna_logs (qna_id, tstp, user_question, response)
            VALUES (:qna_id, :tstp, :user_question, :response);
        """
        )
        conn.execute(
            query,
            {
                "qna_id": str(qna_id),
                "tstp": tstp,
                "user_question": str(user_question),
                "response": str(response),
            },
        )


def load_arxiv():
    query = "SELECT * FROM arxiv_details;"
    conn = create_engine(database_url)
    arxiv_df = pd.read_sql(query, conn)
    arxiv_df.set_index("arxiv_code", inplace=True)
    return arxiv_df


def load_reviews():
    query = "SELECT * FROM summaries;"
    conn = create_engine(database_url)
    summaries_df = pd.read_sql(query, conn)
    summaries_df.set_index("arxiv_code", inplace=True)
    return summaries_df


def load_topics():
    query = "SELECT * FROM topics;"
    conn = create_engine(database_url)
    topics_df = pd.read_sql(query, conn)
    topics_df.set_index("arxiv_code", inplace=True)
    return topics_df


def load_citations():
    query = "SELECT * FROM semantic_details;"
    conn = create_engine(database_url)
    citations_df = pd.read_sql(query, conn)
    citations_df.set_index("arxiv_code", inplace=True)
    citations_df.drop(columns=["paper_id"], inplace=True)
    return citations_df


def get_arxiv_parent_chunk_ids(chunk_ids: list):
    """Get (arxiv_code, parent_id) for a list of (arxiv_code, child_id) tuples."""
    ## ToDo: Improve version param.
    engine = create_engine(database_url)
    with engine.begin() as conn:
        # Prepare a list of conditions for matching pairs of arxiv_code and child_id
        conditions = " OR ".join(
            [
                f"(arxiv_code = '{arxiv_code}' AND child_id = {child_id})"
                for arxiv_code, child_id in chunk_ids
            ]
        )
        query = text(
            f"""
            SELECT DISTINCT arxiv_code, parent_id
            FROM arxiv_chunk_map
            WHERE ({conditions})
            AND version = '10000_1000';
--             AND version = '5000_500';
            """
        )
        result = conn.execute(query)
        parent_ids = result.fetchall()
    return parent_ids


def get_arxiv_chunks(chunk_ids: list, source="child"):
    """Get chunks with metadata for a list of (arxiv_code, chunk_id) tuples."""
    engine = create_engine(database_url)
    source_table = "arxiv_chunks" if source == "child" else "arxiv_parent_chunks"
    with engine.begin() as conn:
        # Prepare a list of conditions for matching pairs of arxiv_code and chunk_id
        conditions = " OR ".join(
            [
                f"(p.arxiv_code = '{arxiv_code}' AND p.chunk_id = {chunk_id})"
                for arxiv_code, chunk_id in chunk_ids
            ]
        )
        query = text(
            f"""
            SELECT d.arxiv_code, d.published, s.citation_count, p.text
            FROM {source_table} p , arxiv_details d, semantic_details s
            WHERE p.arxiv_code = d.arxiv_code
            AND p.arxiv_code = s.arxiv_code
            AND ({conditions});
            """
        )
        result = conn.execute(query)
        chunks = result.fetchall()
        chunks_df = pd.DataFrame(chunks)
    return chunks_df


def check_in_db(arxiv_code, db_params, table_name):
    """Check if an arxiv code is in the database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE arxiv_code = '{arxiv_code}'")
            return bool(cur.rowcount)


def upload_to_db(data, db_params, table_name):
    """Upload a dictionary to a database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["%s"] * len(data))
            cur.execute(
                f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
                list(data.values()),
            )


def remove_from_db(arxiv_code, db_params, table_name):
    """Remove an entry from the database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE arxiv_code = '{arxiv_code}'")


def upload_df_to_db(df, table_name, params, if_exists="append"):
    """Upload a dataframe to a database."""
    db_url = (
        f"postgresql+psycopg2://{params['user']}:{params['password']}"
        f"@{params['host']}:{params['port']}/{params['dbname']}"
    )
    engine = create_engine(db_url)
    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=10,
    )

    ## Commit.
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute("COMMIT")

    ## Close.
    engine.dispose()

    return True


def get_arxiv_id_list(db_params, table_name):
    """Get a list of all arxiv codes in the database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT arxiv_code FROM {table_name}")
            return [row[0] for row in cur.fetchall()]


def get_max_table_date(db_params, table_name, date_col="date"):
    """Get the max date in a table."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX({date_col}) FROM {table_name}")
            return cur.fetchone()[0]


def get_arxiv_id_embeddings(db_params, collection_name):
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT a.cmetadata->>'arxiv_code' AS arxiv_code
                FROM langchain_pg_embedding a, langchain_pg_collection b
                WHERE a.collection_id = b.uuid
                AND b.name = '{collection_name}'
                AND a.cmetadata->>'arxiv_code' IS NOT NULL;"""
            )
            return [row[0] for row in cur.fetchall()]


def get_arxiv_title_dict(db_params=db_params):
    """Get a list of all arxiv titles in the database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
            SELECT a.arxiv_code, a.title 
            FROM arxiv_details a
            RIGHT JOIN summaries s ON a.arxiv_code = s.arxiv_code
            WHERE a.title IS NOT NULL
            """
            )
            title_map = {row[0]: row[1] for row in cur.fetchall()}
            return title_map


def get_topic_embedding_dist(db_params=db_params):
    """ Get mean and stdDev for topic embeddings (dim1 & dim2). """
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
            SELECT AVG(dim1), STDDEV(dim1), AVG(dim2), STDDEV(dim2)
            FROM topics
            """
            )
            res = cur.fetchone()
            res = {
                "dim1": {"mean": res[0], "std": res[1]},
                "dim2": {"mean": res[2], "std": res[3]},
            }
            return res

def get_weekly_summary_inputs(date: str):
    """Get weekly summaries for a given date (from last monday to next sunday)."""
    engine = create_engine(database_url)
    ## Find last monday if not monday.
    date_st = pd.to_datetime(date).date() - pd.Timedelta(
        days=pd.to_datetime(date).weekday()
    )
    ## Find next sunday if not sunday.
    date_end = pd.to_datetime(date).date() + pd.Timedelta(
        days=6 - pd.to_datetime(date).weekday()
    )
    with engine.begin() as conn:
        query = text(
            f"""
            SELECT d.published, d.arxiv_code, d.title, d.authors, sd.citation_count, d.arxiv_comment,
                   d.summary, s.contribution_content, s.takeaway_content, s.takeaway_example
            FROM summaries s, arxiv_details d, semantic_details sd
            WHERE s.arxiv_code = d.arxiv_code 
            AND s.arxiv_code = sd.arxiv_code
            AND d.published BETWEEN '{date_st}' AND '{date_end}'
            """
        )
        result = conn.execute(query)
        summaries = result.fetchall()
        summaries_df = pd.DataFrame(summaries)
    return summaries_df


def check_weekly_summary_exists(date_str: str):
    """Check if weekly summary exists for a given date."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        query = text(
            f"""
            SELECT COUNT(*)
            FROM weekly_reviews
            WHERE date = '{date_str}'
            """
        )
        result = conn.execute(query)
        count = result.fetchone()[0]

    engine.dispose()
    return count > 0


def get_weekly_summary(date_str: str):
    """Get weekly summary for a given date."""
    engine = create_engine(database_url)
    date_str = (
        pd.to_datetime(date_str).date()
        - pd.Timedelta(days=pd.to_datetime(date_str).weekday())
    ).strftime("%Y-%m-%d")
    with engine.begin() as conn:
        query = text(
            f"""
            SELECT review
            FROM weekly_reviews
            WHERE date = '{date_str}'
            """
        )
        result = conn.execute(query)
        review = result.fetchone()[0]

    engine.dispose()
    return review
