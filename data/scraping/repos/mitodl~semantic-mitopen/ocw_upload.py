import argparse
import hashlib
import os
import re
import sys
import time
from math import ceil
from threading import Thread

import numpy as np
import openai
import psycopg2
import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from pgvector.psycopg2 import register_vector
from psycopg2.extras import RealDictCursor

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))
CHUNK_MAX = CHUNK_SIZE + 50


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class ContentVector:
    def __init__(
        self,
        content_file,
        title,
        url,
        content,
        content_length,
        content_tokens,
        embedding,
    ):
        self.run_title = content_file["run_title"]
        self.platform = content_file["platform"]
        self.run_id = content_file["run_id"]
        self.run_key = content_file["run_key"]
        self.run_url = content_file["run_url"]
        self.content_id = content_file["id"]
        self.platform = content_file["platform"]
        self.content_title = title
        self.content_url = url
        self.content = content
        self.content_hash = get_hash(content_file, content)
        self.content_length = content_length
        self.content_tokens = content_tokens
        self.embedding = embedding


def get_title(content_file):
    return (
        content_file["content_title"]
        or content_file["title"]
        or content_file["key"].split("/")[-1]
    )


def get_url(content_file):
    if content_file["key"]:
        return f'https://ocw.mit.edu/{content_file["key"]}'


def get_hash(content_file, content):
    return hashlib.md5(
        f'{content_file["platform"]}_{content_file["run_key"]}_{content}'.encode(
            "utf-8"
        )
    ).hexdigest()


def get_content(content_file):
    lines = [
        f"@@@^^^{line.strip()}"
        for line in re.sub(r"[^\s\w\.]+", "", content_file["content"]).split("\n")
        if line.strip() != ""
    ]
    if len(lines) > 0:
        lines = " ".join(lines)
        return lines
    else:
        return None


def chunk_file_by_sections(content):
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return [chunk.page_content for chunk in splitter.create_documents([content])]


def chunk_file_by_size(content):
    page_text_chunks = []
    if num_tokens_from_string(content, "cl100k_base") > CHUNK_SIZE:
        split = "@@@^^^".join(content.split(". ")).split("@@@^^^")
        chunkText = ""
        for sentence in split:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            sentence_tokens = num_tokens_from_string(sentence, "cl100k_base")
            if sentence_tokens > CHUNK_SIZE:
                continue
            chunk_tokens = num_tokens_from_string(chunkText, "cl100k_base")
            if chunk_tokens + sentence_tokens > CHUNK_SIZE:
                page_text_chunks.append(chunkText.strip())
                chunkText = ""
            if re.search("[a-zA-Z]", sentence[-1]):
                chunkText += sentence + ". "
            else:
                chunkText += sentence + " "
        page_text_chunks.append(chunkText.strip())
    else:
        page_text_chunks.append(content.strip())

    if len(page_text_chunks) > 2:
        last_elem = num_tokens_from_string(page_text_chunks[-1], "cl100k_base")
        second_to_last_elem = num_tokens_from_string(
            page_text_chunks[-2], "cl100k_base"
        )
        if last_elem + second_to_last_elem < CHUNK_MAX:
            page_text_chunks[-2] += page_text_chunks[-1]
            page_text_chunks.pop()

    return page_text_chunks


def embed_chunk(resource, title, url, content):
    embedding = openai.Embedding.create(input=content, model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]
    chunk = ContentVector(
        resource,
        title,
        url,
        content,
        len(content),
        num_tokens_from_string(content, "cl100k_base"),
        embedding,
    )
    return chunk


def make_file_embeddings(cursor, content_file, delete_existing=False):
    title = get_title(content_file)
    content = get_content(content_file)
    #content = content_file["content"].strip()
    url = get_url(content_file)

    if delete_existing:
        print("Deleting old chunks...")
        # Delete any existing chunks for this file
        cursor.execute(
            "DELETE FROM "
            + os.getenv("POSTGRES_TABLE_NAME")
            + " WHERE content_id = %s",
            (content_file["id"],),
        )
    else:
        # Skip processing if we already have a chunk for this file
        cursor.execute(
            "SELECT content_id FROM "
            + os.getenv("POSTGRES_TABLE_NAME")
            + " WHERE content_id = %s",
            (content_file["id"],),
        )
        row = cursor.fetchone()
        if row:
            print(f"Skipping, existing chunk for {content_file['key']}")
            return False

    if not content:
        return False
    page_text_chunks = chunk_file_by_size(content)
    print(f"Chunked into {len(page_text_chunks)} sections")
    for chunk in page_text_chunks:
        try:
            pg_chunk = embed_chunk(content_file, title, url, chunk)
        except:
            print("Embed API request failed, trying again in 5 seconds...")
            time.sleep(5)
            try:
                pg_chunk = embed_chunk(content_file, title, url, chunk)
            except Exception as e:
                print(f"Failed to embed {content_file['title']}")
                print(e)
                return
        embedding = np.array(pg_chunk.embedding)
        sql = (
            "INSERT INTO "
            + os.getenv("POSTGRES_TABLE_NAME")
            + "(run_title, run_id, run_key, run_url, platform, page_title, content_title, page_url, content_url, content, content_id, content_hash, content_length, content_tokens, embedding)"
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        )
        #'ON CONFLICT(content_hash) DO UPDATE SET embedding = %s;'
        cursor.execute(
            sql,
            (
                pg_chunk.run_title,
                pg_chunk.run_id,
                pg_chunk.run_key,
                pg_chunk.run_url,
                pg_chunk.platform,
                pg_chunk.content_title,
                pg_chunk.content_title,
                pg_chunk.content_url,
                pg_chunk.content_url,
                pg_chunk.content,
                pg_chunk.content_id,
                pg_chunk.content_hash,
                str(pg_chunk.content_length),
                str(pg_chunk.content_tokens),
                # embedding,
                embedding,
            ),
        )
    return True


def process_courses(course_ids, delete_existing=False):
    conn_open_batch = None
    conn_vector_batch = None

    try:
        print(f"Processing {len(course_ids)} courses")
        conn_open_batch = psycopg2.connect(
            host=os.getenv("OPEN_POSTGRES_HOST"),
            database=os.getenv("OPEN_POSTGRES_DB_NAME"),
            user=os.getenv("OPEN_POSTGRES_USERNAME"),
            password=os.getenv("OPEN_POSTGRES_PASSWORD"),
            cursor_factory=RealDictCursor,
        )

        conn_open_cursor = conn_open_batch.cursor()

        conn_vector_batch = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DB_NAME"),
            user=os.getenv("POSTGRES_USERNAME"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        register_vector(conn_vector_batch)
        conn_vector_cursor = conn_vector_batch.cursor()

        OPEN_QUERY = """
        DECLARE super_cursor CURSOR FOR SELECT cf.id, cf.key, cf.title, cf.content, cf.content_title, cf.url, run.title as run_title, run.id as run_id, run.platform as platform, run.run_id as run_key,
        run.url as run_url, run.platform as platform, course.course_id FROM course_catalog_contentfile as cf
        LEFT JOIN course_catalog_learningresourcerun AS run ON cf.run_id = run.id INNER JOIN course_catalog_course AS course ON run.object_id = course.id
        WHERE cf.content IS NOT NULL and cf.content != '' and course.published IS TRUE and run.published IS TRUE and course.course_id IN %s ORDER BY course.course_id ASC, run.run_id ASC, cf.id ASC;
        """

        print("Getting content files...")
        conn_open_cursor.execute(OPEN_QUERY, [tuple(course_ids)])
        course = None
        run = None
        still_processing = True
        while still_processing:
            conn_open_cursor.execute("FETCH 10 FROM super_cursor")
            content_files = conn_open_cursor.fetchall()
            still_processing = len(content_files) > 0
            for content_file in content_files:
                if not content_file["content"].strip():
                    continue
                if content_file["course_id"] != course:
                    print(f"Course: {content_file['course_id']}")
                    course = content_file["course_id"]
                if content_file["run_id"] != run:
                    print(f"(Run: {content_file['run_id']})")
                    run = content_file["run_id"]
                print(f"Embedding {content_file['key']}")
                make_file_embeddings(conn_vector_cursor, content_file, delete_existing)
                print("Committing...")
                conn_vector_batch.commit()
        print("Done embedding files for this batch of courses.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        raise error
    finally:
        if conn_vector_batch is not None:
            conn_vector_batch.close()
        if conn_open_batch is not None:
            conn_open_batch.close()
        print(f"Done processing {course_ids}")
        return


def chunks(ids, num_chunks):
    size = ceil(len(ids) / num_chunks)
    return list(map(lambda x: ids[x * size : x * size + size], list(range(num_chunks))))


def main():

    parser = argparse.ArgumentParser(
        description="Create embeddings for MIT Open course content files."
    )
    parser.add_argument(
        "--threads",
        dest="threads",
        type=int,
        default=5,
        help="Number of simultaneous threads to run",
    )
    parser.add_argument(
        "--ids",
        dest="course_id_filter",
        nargs="*",
        default=[],
        help="list of course_ids to process",
    )
    parser.add_argument(
        "--delete",
        dest="delete_existing",
        default=False,
        action="store_true",
        help="Delete existing embeddings for each content file",
    )

    args = parser.parse_args()

    course_id_filter = args.course_id_filter
    print(f"COURSE ID FILTER: {course_id_filter}")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    conn_open = None

    try:
        conn_open = psycopg2.connect(
            host=os.getenv("OPEN_POSTGRES_HOST"),
            database=os.getenv("OPEN_POSTGRES_DB_NAME"),
            user=os.getenv("OPEN_POSTGRES_USERNAME"),
            password=os.getenv("OPEN_POSTGRES_PASSWORD"),
            cursor_factory=RealDictCursor,
        )

        cur_open = conn_open.cursor()

        OPEN_QUERY = """
        SELECT DISTINCT course_id from course_catalog_course WHERE published IS TRUE and platform = 'ocw' ORDER BY course_id DESC;
        """
        query_args = [OPEN_QUERY]

        if course_id_filter:
            query_args = [OPEN_QUERY.replace("WHERE", "WHERE course_id IN %s AND ")]
            query_args.append([tuple(course_id_filter)])

        print("Getting content files...")
        cur_open.execute(*query_args)

        course_ids = [result["course_id"] for result in cur_open.fetchall()]

        print(f"Processing {len(course_ids)} courses")

        # Divide the content_files into 5 chunks
        threads = []
        for chunk in chunks(course_ids, args.threads):
            thread = Thread(
                target=process_courses, args=([chunk, args.delete_existing])
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    except (Exception, psycopg2.DatabaseError) as error:
        raise error
    finally:
        if conn_open is not None:
            conn_open.close()
            print("MIT Open database connection closed.")


if __name__ == "__main__":
    main()
