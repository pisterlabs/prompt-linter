#!/usr/local/bin/python

import requests
import openai
import json
from datetime import datetime
import os

import io
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from config import OPENAI_API_KEY
import asyncpg
import asyncio
import logging

pool = None
os.environ["STABILITY_HOST"] = "grpc.stability.ai:443"

openai.api_key = OPENAI_API_KEY


def setup_logging(level=logging.DEBUG):
    """
    Set up and configure a logging object.

    Args:
        level (int, optional): The log level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: A configured logging object.
    """
    environment = os.environ.get("APP_ENVIRONMENT", "dev")
    if not level:
        if environment == "prod":
            level = logging.INFO
        else:
            level = logging.DEBUG

    # Create a logging object
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    log_file = "logs/bot-e.log"

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a file handler (to write logs to a file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


def debug(msg):
    logger.debug(msg)


def warn(msg):
    logger.warn(msg)


def critical(msg):
    logger.critical(msg)


def info(msg):
    logger.info(msg)


async def create_db_pool():
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(
            host="localhost",
            database="bot-e",
            port=5432,
        )
    return pool


async def db_connect2():
    pool = await create_db_pool()
    connection = await pool.acquire()

    return connection


async def db_release(connection):
    # Release the connection back to the pool
    await pool.release(connection)


async def new_question(conn, question, session_id):
    question = question.strip()
    if len(question) < 1 or len(session_id) < 1:
        return {}

    async with conn.transaction():
        result = await conn.fetchrow(
            "INSERT INTO question (question, creator_session_id) VALUES ($1, $2) RETURNING question_id, question, creator_session_id",
            question,
            session_id,
        )

    return dict(result)


async def answer_next(conn):
    async with conn.transaction():
        result = await conn.fetchrow(
            "SELECT * FROM question WHERE answer IS NULL ORDER BY added_at ASC LIMIT 1;"
        )
    if result:
        return dict(result)
    else:
        return None


async def annotate_question(conn, question):
    moderation = moderation_api(question["question"])
    embedding = embedding_api(question["question"])
    async with conn.transaction():
        await conn.execute(
            "UPDATE question SET moderation = $1, embedding = $2 WHERE question_id = $3",
            json.dumps(moderation),
            json.dumps(embedding),
            question["question_id"],
        )


async def post_question(question, session_id):
    try:
        conn = await db_connect2()
        question_record = await new_question(conn, question, session_id)
        await annotate_question(conn, question_record)
        return question_record
    finally:
        await db_release(conn)


def moderation_api(input_text):
    response = openai.Moderation.create(input=input_text)
    return response


def validate_key(key):
    # Define the set of allowed characters
    allowed_chars = set(
        "-_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )

    # Check if the string is 11 characters long
    if len(key) != 11:
        return False

    # Check if the first or last character is _ or -
    if key[0] in "-_" or key[-1] in "-_":
        return False

    # Check if all characters are in the allowed set
    for char in key:
        if char not in allowed_chars:
            return False

    # If all tests pass, return True
    return True


async def search(search_for):
    conn = await db_connect2()

    try:
        query = """
            SELECT question_id, question, answer, full_image_url(image_url) AS image_url, media, title, description, TO_CHAR(added_at, 'YYYY-MM-DD HH24:MI:SS') AS added_at FROM search($1)
        """
        questions = await conn.fetch(query, search_for)

        data = [dict(row) for row in questions]
        return data
    finally:
        await db_release(conn)


async def trending(start_date):
    conn = await db_connect2()

    try:
        query = """
            SELECT * FROM get_top_upvotes($1)
        """
        questions = await conn.fetch(query, start_date)

        data = [dict(row) for row in questions]

        return data
    finally:
        await db_release(conn)


async def row_lock(lock_key):
    try:
        conn = await db_connect2()

        query = """
            SELECT lock_row($1)
            """

        async with conn.transaction():
            lock = await conn.fetchrow(query, lock_key)

        lo = lock["lock_row"]
        if not lock["lock_row"]:
            return False

        return True
    finally:
        await db_release(conn)


async def get_question(question_id, simplified=False):
    """
    suface all question data
    """
    try:
        conn = await db_connect2()

        if not validate_key(question_id):
            return json.dumps([])

        if simplified:
            query = """
                SELECT 
                    question_id, 
                    question, 
                    answer, 
                    full_image_url(image_url) AS image_url, 
                    media, 
                    title, 
                    description, 
                    creator_session_id,
                    TO_CHAR(added_at, 'YYYY-MM-DD HH24:MI:SS') AS added_at
                FROM question
                WHERE question_id = $1
                """
        else:
            query = """
                SELECT *
                FROM question
                WHERE question_id = $1
                """

        async with conn.transaction():
            question = await conn.fetchrow(query, question_id)

        if not question:
            return json.dumps([])

        data = dict(question)
        return data
    finally:
        await db_release(conn)


async def simplified_question(question_id):
    """
    suface limited data to api
    """
    return await get_question(question_id, simplified=True)


async def question_comments(question_id):
    # TODO: moderate comments
    conn = await db_connect2()

    if not validate_key(question_id):
        return json.dumps([])

    query = """
        SELECT
            question_id,
            comment,
            session_id,
            parent_comment_id,
            comment_id,
            TO_CHAR(added_at, 'YYYY-MM-DD HH:MI AM') AS added_at
        FROM question_comment
        WHERE question_id = $1
        ORDER BY added_at DESC
        LIMIT 100
        """
    try:
        async with conn.transaction():
            comments = await conn.fetch(query, question_id)

        if not comments:
            return []

        comments_list = [dict(comment) for comment in comments]

        return comments_list
    except Exception:
        return json.dumps([])
    finally:
        await db_release(conn)


async def random_question():
    conn = await db_connect2()

    try:
        query = "SELECT * FROM question ORDER BY RANDOM() LIMIT 1"
        question = await conn.fetchrow(query)  # fetchrow returns a single record

        if not question:
            return []

        return dict(question)  # Convert the record to a dictionary

    except Exception:
        return []
    finally:
        await conn.close()


async def next_question(question_id, session_id, direction):
    conn = await db_connect2()  # Assuming db_connect2 is your async connection function

    if not validate_key(question_id):
        return "[]"

    if direction == "down":
        similarity = False
    else:
        similarity = True

    try:
        query = "select * from proximal_question($1, $2, $3)"
        question = await conn.fetchrow(query, question_id, session_id, similarity)

        if not question:
            return []

        return dict(question)  # Convert the record to a dictionary

    except Exception:
        return []
    finally:
        await db_release(conn)


def embedding_api(input_text):
    response = openai.Embedding.create(input=input_text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]


async def respond(question_id):
    logger.debug(f"beginning respond function: {question_id}")
    if not validate_key(question_id):
        return {"error": "invalid question_id"}
    question = await simplified_question(question_id)
    if not question:
        return {"error": "invalid question_id"}

    try:
        conn = await db_connect2()
        question = await respond_to_question(conn, question)
        logger.debug(f"response function, response recieved: {question_id}")
        return question
    finally:
        await db_release(conn)


async def respond_to_question(conn, data):
    user_message = data["question"]
    with open("data/system_prompt.txt", "r") as file:
        system_prompt = file.read()

    question_id = data["question_id"]

    logger.debug(f"beginning response_to_question {question_id}")

    system_message = f"{system_prompt}"

    messages = [
        {
            "role": "system",
            "content": f"{system_message}",
        },
        {
            "role": "user",
            "content": f"{user_message}",
        },
    ]

    logger.debug("begin openai chat completion")

    completion = await asyncio.to_thread(
        openai.ChatCompletion.create, model="gpt-4", messages=messages
    )

    logger.debug("received openai chat completion")

    messages.append(completion["choices"][0]["message"])

    response_message = completion["choices"][0]["message"]["content"]

    # Parse the response message
    lines = response_message.split("\n", 1)
    title_line = lines[0]
    rest_of_answer = lines[1]

    # if this fails, the model did not return a title
    try:
        title = title_line.split(":", 1)[1].strip()
    except IndexError:
        logger.warn("title not recieved in chat completion")
        title = ""
        rest_of_answer = response_message

    logger.debug("begin saving data")
    async with conn.transaction():
        # Execute SQL statements using asyncpg
        await conn.execute(
            "update question set answer = $1, system_prompt = $2, title = $3 where question_id = $4",
            json.dumps(rest_of_answer),
            system_message,
            title,
            question_id,
        )
    return await simplified_question(question_id)


async def enrich_question(question_id):
    question = await get_question(question_id)
    system_message = question["system_prompt"]
    user_message = question["question"]
    assistant_message = question["answer"]
    title = question["title"]

    debug(title)
    if not assistant_message:
        return {"error": "questions must be answered before they can be enriched."}

    messages = [
        {
            "role": "system",
            "content": f"{system_message}",
        },
        {
            "role": "user",
            "content": f"{user_message}",
        },
        {
            "role": "assistant",
            "content": f"{assistant_message}",
        },
    ]

    with open("data/question_functions.json", "r") as file:
        functions = json.load(file)

    logger.debug("begin function call to openAI")
    function_message = await asyncio.to_thread(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=messages,
        functions=functions,
        function_call={"name": "extract_data"},
    )

    function_message = function_message["choices"][0]["message"]

    if function_message.get("function_call"):
        function_response = json.loads(function_message["function_call"]["arguments"])
        description = function_response["description"]
        media = function_response["media"]

        if len(title) == 0:
            # if there was a previous title failure....
            title = function_response["title"]

        try:
            logger.debug("begin stability ai image generation")
            image_url = await asyncio.to_thread(stability_image, title, question_id)
        except Exception:
            logger.debug("begin dall-e image generation")
            image_url = await asyncio.to_thread(openai_image, title, question_id)

        logger.debug("begin saving enrichment data")
        try:
            conn = await db_connect2()
            await conn.execute(
                "update question set title = $1, description = $2, image_url = $3, media = $4 where question_id = $5",
                title,
                description,
                image_url,
                json.dumps(media),
                question_id,
            )
        finally:
            await db_release(conn)

    logger.info(f"bot-e enrichment response complete for {question_id}")
    return await simplified_question(question_id)


def openai_image(title, question_id):
    try:
        response = openai.Image.create(
            prompt=f"Generate an abstract image representing: {title}",
            n=1,
            size="512x512",
        )
        image_url = response["data"][0]["url"]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(base_dir, "images")
        questions_dir = os.path.join(images_dir, "questions")
        folder_character = question_id[0].lower()
        question_subfolder = os.path.join(questions_dir, folder_character)
        os.makedirs(question_subfolder, exist_ok=True)
        image_data = requests.get(image_url).content
        image_filename = os.path.join(question_subfolder, f"{question_id}.png")
        with open(image_filename, "wb") as image_file:
            image_file.write(image_data)
        return f"/images/questions/{folder_character}/{question_id}.png"
    except openai.error.InvalidRequestError:
        image_url = ""
    return image_url


def stability_image(title, question_id):
    logger.debug("begin stability_image function")
    stability_api = client.StabilityInference(
        key=os.environ["STABILITY_KEY"],
        verbose=True,  # Print debug messages.
        engine="stable-diffusion-xl-1024-v1-0",
    )

    logger.debug("stability api initialized")
    answers = stability_api.generate(
        prompt=f"using two complimentary colors, create an image that will make people curious about this title: '{title}'. ",
        seed=4253978046,
        steps=30,
        cfg_scale=7.0,
        width=512,
        height=512,
        samples=1,
        sampler=generation.SAMPLER_K_DPMPP_2M,
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                raise ValueError(
                    "Your request activated the API's safety filters and could not be processed."
                )
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                base_dir = os.path.dirname(os.path.abspath(__file__))
                images_dir = os.path.join(base_dir, "images")
                questions_dir = os.path.join(images_dir, "questions")
                folder_character = question_id[0].lower()
                question_subfolder = os.path.join(questions_dir, folder_character)
                os.makedirs(question_subfolder, exist_ok=True)
                image_filename = os.path.join(question_subfolder, f"{question_id}.png")

                img.save(image_filename)
                os.chmod(image_filename, 0o664)
                return f"/images/questions/{folder_character}/{question_id}.png"

    return ""


if __name__ == "__main__":
    asyncio.run(row_lock("corn"))
