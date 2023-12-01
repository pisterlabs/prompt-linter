import datetime
import os
import random
from functools import lru_cache
from typing import List

import numpy as np
import pytz as pytz
import shelve
from cache import AsyncLRU
from discord import Message
from tqdm.auto import tqdm

from data import HAGRID_COMMANDS
from openai_utils import generate_embedding, generate_text, num_tokens_from_string
from stats import stat

os.makedirs("shelve/", exist_ok=True)

progress = shelve.open("shelve/progress")
settings = shelve.open("shelve/settings")

import sqlite3

con = sqlite3.connect("shelve/database.db")

SUPER_SMART = False


def setup():
    con.execute(
        """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY,
        guild INTEGER,
        channel INTEGER,
        author INTEGER,
        content TEXT,
        date DATETIME,
        indexed BOOLEAN
    )
    """
    )

    con.execute(
        """
    CREATE TABLE IF NOT EXISTS embeddings (
        message_id INTEGER PRIMARY KEY,
        embedding BLOB
    )
    """
    )

    con.execute(
        """
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        guild INTEGER,
        channel INTEGER,
        from_date DATETIME,
        to_date DATETIME,
        summary TEXT,
        embedding BLOB
    )
    """
    )

    con.execute(
        """
    CREATE TABLE IF NOT EXISTS names (
        id INTEGER PRIMARY KEY,
        name TEXT
    )
    """
    )

    con.execute(
        """
    CREATE TABLE IF NOT EXISTS blocked_users (
        id INTEGER PRIMARY KEY,
        reasons TEXT
    )
    """
    )


setup()


@lru_cache(maxsize=1024)
def set_name(identifier: int, name: str) -> None:
    con.execute(
        "INSERT OR REPLACE INTO names (id, name) VALUES (?, ?)", (identifier, name)
    )


@lru_cache(maxsize=1024)
def get_name(identifier: int) -> None:
    name = con.execute("FETCH name FROM names WHERE id =?", (identifier,)).fetchone()
    return str(identifier) if name is None else name


MIN_EMBEDDING_LENGTH = 32
MAX_CONVERSATION_TIME = 45

active_conversations = {}


def get_all_embeddings(guild_id: int) -> (np.array, List[str]):
    """
    :param guild_id: The current guild
    :return: The embedding matrix and the jump urls for all indexed messages
    """
    results = con.execute(
        """
        SELECT messages.id, content, embeddings.embedding, channel, names.name
        FROM messages
        LEFT JOIN embeddings ON messages.id=embeddings.message_id
        LEFT JOIN names ON messages.channel=names.id
        LEFT JOIN blocked_users ON messages.author=blocked_users.id
        WHERE blocked_users.id IS NULL AND indexed = TRUE AND guild = ? AND length(content) > ?
    """,
        (
            guild_id,
            MIN_EMBEDDING_LENGTH,
        ),
    ).fetchall()

    embeddings = []
    messages = []

    for result in tqdm(results, "Embedding messages"):
        # Fetch missing embeddings
        if result[2] is None:
            embedding = generate_embedding(result[4] + ": " + result[1])
            con.execute(
                "INSERT INTO embeddings (message_id, embedding) VALUES (?, ?)",
                (result[0], embedding.tobytes()),
            )
        else:
            embedding = np.frombuffer(result[2], dtype=np.float32)
        embeddings.append(embedding)

        messages.append(
            f"https://discord.com/channels/{guild_id}/{result[3]}/{result[0]}"
        )

    con.commit()

    return np.asarray(embeddings), messages


def get_cosine_similarity(arr_a: np.array, arr_b: np.array) -> np.array:
    """
    Returns the similarity between all pairs between arr_a and arr_b
    :param arr_a: A matrix of shape (n_samples, dim)
    :param arr_b: A matrix of shape (n_samples, dim)
    :return: A similarity matrix of shape (n_samples_a, n_samples_b)
    """
    norms_a = np.linalg.norm(arr_a, axis=-1)[:, np.newaxis]
    norms_b = np.linalg.norm(arr_b, axis=-1)[:, np.newaxis]
    divisor = norms_a * norms_b.T
    dot_p = arr_a @ arr_b.T
    return np.divide(dot_p, divisor, dot_p, where=divisor > 0)


def search(guild_id: int, embedding: np.array, samples: int = 3) -> List[str]:
    """
    Searches for jump urls to similar messages to the given embedding
    :param guild_id: Current guild
    :param embedding: Embedding vector
    :param samples: Max samples to look for
    :return: A list of jump urls
    """
    embeddings, messages = get_all_embeddings(guild_id)

    similarities = get_cosine_similarity(embeddings, embedding[None, :]).flatten()

    best_guess = similarities.max()

    top_ids = np.argsort(-similarities)[:samples]

    return [messages[i] for i in top_ids if similarities[i] > best_guess * 0.95]


def find_best_summaries(
        guild_id: int, embedding: np.array, samples: int = 1
) -> List[str]:
    """
    Searches for summaries most similar to a user query
    :param guild_id: Current guild
    :param embedding: Embedding vector
    :param samples: Max samples to look for
    :return: A list of jump urls
    """
    results = con.execute(
        """
        SELECT summary, embedding
        FROM summaries
        WHERE guild = ?
    """,
        (guild_id,),
    ).fetchall()

    embeddings = np.asarray([np.frombuffer(r[1], dtype=np.float32) for r in results])

    similarities = get_cosine_similarity(embeddings, embedding[None, :]).flatten()

    best_guess = similarities.max()

    top_ids = np.argsort(-similarities)[:samples]

    return [results[i][0] for i in top_ids if similarities[i] > best_guess * 0.75]


def drop_until(messages: List[str], max_size: int, encoding_name="gpt-3.5-turbo"):
    while (
            len(messages) > 0
            and num_tokens_from_string("\n".join(messages), encoding_name) > max_size
    ):
        messages.pop(random.randrange(len(messages)))
    return messages


@AsyncLRU(3600)
async def who_is(user_id, max_length=3000):
    messages = con.execute(
        "SELECT content FROM messages WHERE author=? ORDER BY RANDOM() LIMIT 1000",
        (user_id,),
    ).fetchall()

    messages = [m[0] for m in messages if not m[0].startswith("/")]
    drop_until(messages, max_length)

    prompt = "> " + "\n> ".join(messages)

    print(f"Describing user with {len(messages)} messages and {len(prompt)} chars")

    system_prompt = "You are a language model tasked with describing this person in a few honest sentences, based on their past messages. Put focus on personality and behavior."
    return await generate_text(prompt, system_prompt=system_prompt, max_tokens=256)


def get_yesterday_boundary() -> (datetime, datetime):
    # Define the CEST timezone
    cest = pytz.timezone("Europe/Berlin")  # Central European Summer Time (CEST)

    # Get the current date and time in the CEST timezone
    now_cest = datetime.datetime.now(cest)

    # Set the time to 12:00 PM (noon)
    noon_cest = now_cest.replace(hour=12, minute=0, second=0, microsecond=0)

    # If the current time is before 12:00 PM, consider yesterday's boundary
    if now_cest < noon_cest:
        yesterday = noon_cest - datetime.timedelta(days=1)
    else:
        yesterday = noon_cest

    return yesterday - datetime.timedelta(days=1), yesterday


@AsyncLRU(3)
async def get_summary(guild_id, channel_id, offset: int = 0, max_length: int = 3500):
    from_date, to_date = get_yesterday_boundary()
    from_date = from_date - datetime.timedelta(days=offset)
    to_date = to_date - datetime.timedelta(days=offset)

    summary = con.execute(
        "SELECT summary FROM summaries WHERE guild=? AND channel=? AND from_date=? AND to_date=?",
        (guild_id, channel_id, from_date, to_date),
    ).fetchone()

    if summary is None:
        messages = con.execute(
            """
            SELECT content, names.name as username
            FROM messages
            LEFT JOIN names ON names.id=messages.author
            LEFT JOIN blocked_users ON messages.author=blocked_users.id
            WHERE blocked_users.id IS NULL AND guild=? AND (? < 0 OR channel=?) AND date BETWEEN ? AND ?
            """,
            (guild_id, channel_id, channel_id, from_date, to_date),
        ).fetchall()

        messages = [f"{m[1]}: {m[0]}" for m in messages if not m[0].startswith("/")]

        original_count = len(messages)

        # Crop to fit into context size
        messages = drop_until(messages, max_length)

        # Construct prompt
        prompt = "> " + "\n> ".join(messages)

        # No need to summarize a single message
        if len(prompt) < 64:
            return "Nothing.", from_date, to_date

        prompt = f"Today's new conversations:{prompt}\n\nToday's summary"

        # Generate summary
        where = (
            "public Discord server"
            if channel_id < 0
            else "specific Discord server channel"
        )
        system_prompt = f"You are a language model tasked with summarizing following conversation on a {where} in a concise way."
        summary = await generate_text(
            prompt,
            system_prompt=system_prompt,
            max_tokens=256,
            model="gpt-3.5-turbo",
            temperature=0.25,
        )

        # Embed the summary for faster indexing
        embedding = generate_embedding(summary)

        # Cache summary
        con.execute(
            """
            INSERT INTO summaries (guild, channel, from_date, to_date, summary, embedding) 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (guild_id, channel_id, from_date, to_date, summary, embedding),
        )
        con.commit()

        print(
            f"Summarized day with {original_count} original, {len(messages)} compressed messages and {len(prompt)} chars."
        )

        return summary, from_date, to_date
    else:
        return summary[0], from_date, to_date


def set_index_status(channel_id, active: bool):
    if active:
        settings[str(active)] = True
    else:
        del settings[str(active)]

    con.execute(
        """
    UPDATE messages
    SET indexed=?
    WHERE channel=?
    """,
        (1 if active else 0, channel_id),
    )


async def track(message: Message):
    after = None
    if str(message.channel.id) in progress:
        after = progress[str(message.channel.id)]

    indexed = str(message.channel.id) in settings

    set_name(message.guild.id, message.guild.name)
    set_name(message.channel.id, message.channel.name)

    count = 0
    async for received in message.channel.history(
            limit=100, after=after, oldest_first=True
    ):
        if received.clean_content and not received.clean_content.startswith("/hagrid"):
            set_name(received.author.id, received.author.name)

            count += 1

            con.execute(
                "INSERT OR REPLACE INTO messages (id, guild, channel, author, content, date, indexed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    received.id,
                    received.guild.id,
                    received.channel.id,
                    received.author.id,
                    received.clean_content,
                    received.created_at,
                    indexed,
                ),
            )

        progress[str(message.channel.id)] = received.created_at
    con.commit()

    print(
        f"Tracked {count} {('indexed ' if indexed else '')}messages in {message.channel.name} until {progress[str(message.channel.id)]}"
    )

    return count


async def on_smart_message(message):
    msg = message.clean_content.lower()

    tracked = str(message.guild.id) in settings

    # Basic commands
    if msg.startswith("/hagrid"):
        stat(message, "command")
        if msg == "/hagrid guild extend":
            settings[str(message.guild.id)] = True
            await message.delete()
        elif msg == "/hagrid channel index":
            set_index_status(message.channel.id, True)
            if hasattr(message.channel, "parent"):
                set_index_status(message.channel.parent.id, True)
            await message.delete()
        elif msg == "/hagrid channel noindex":
            set_index_status(message.channel.id, False)
            if hasattr(message.channel, "parent"):
                set_index_status(message.channel.parent.id, False)
            await message.delete()
        elif msg == "/hagrid scan":
            await message.delete()
            while await track(message) > 0:
                pass
        else:
            await message.channel.send(HAGRID_COMMANDS)

    # Track messages
    if tracked:
        await track(message)
    else:
        return

    # Search for stuff
    if msg.startswith("hagrid search"):
        stat(message, "search")

        await message.channel.typing()

        if message.reference and message.reference.resolved:
            embedding = generate_embedding(message.reference.resolved.clean_content)
        else:
            embedding = generate_embedding(
                message.clean_content.replace("hagrid search", "")
            )

        best = search(message.guild.id, embedding)
        await message.channel.send("Check out those links: " + (" ".join(best)))

        return True

    # Summarize someones personality
    if "hagrid who is" in msg:
        await message.channel.typing()

        if len(message.mentions) == 0:
            await message.channel.send(f"Yer have to mention someone!")
        else:
            who = message.mentions[0].id
            description = await who_is(who)
            await message.channel.send(description)

        return True

    # Summarize a timespan
    if msg.startswith("hagrid what happened"):
        stat(message, "summary")

        await message.channel.typing()

        here = msg.startswith("hagrid what happened here")

        try:
            history_length = int(
                msg.replace(
                    "hagrid what happened here" if here else "hagrid what happened", ""
                ).strip()
            )
        except ValueError:
            history_length = 1

        history_length = min(7, history_length)

        # Get the summary of the last 3 days
        summaries = [
            get_summary(message.guild.id, message.channel.id if here else -1, i)
            for i in range(history_length)
        ]
        summaries.reverse()

        msg = (
                  ""
                  if history_length == 1
                  else "Right then, 'ere's the summary o' the last few days!\n"
              ) + "\n\n".join(
            [
                f'**{to_date.strftime("%Y, %d %B")}:**\n{summary}'
                for (summary, from_date, to_date) in summaries
            ]
        )

        # noinspection SpellCheckingInspection
        if len(msg) > 1500:
            for (summary, from_date, to_date) in summaries:
                await message.channel.send(
                    f'**{to_date.strftime("%Y, %d %B")}:**\n{summary}'
                )
        else:
            await message.channel.send(msg)

        return True

    convo_id = f"{message.author.id}_{message.channel.id}"

    if msg == "bye hagrid" and convo_id in active_conversations:
        del active_conversations[convo_id]
        return True

    if "hallo hagrid" in msg or (
            convo_id in active_conversations
            and (datetime.datetime.now() - active_conversations[convo_id]).seconds
            < MAX_CONVERSATION_TIME
    ):
        stat(message, "hallo hagrid")

        await message.channel.typing()

        FIXED_HISTORY_N = 5
        MAX_HISTORY_N = 32
        WHO_IS_CONTEXT_LENGTH = 512
        HISTORY_LENGTH = 512

        active_conversations[convo_id] = datetime.datetime.now()

        # We are not interested in this specific summary, but let's enforce generating it for the lookup
        if SUPER_SMART:
            await get_summary(message.guild.id, -1)
            await get_summary(message.guild.id, message.channel.id)

            # Fetch the embedding of the input text to look for similar topics
            embedding = await generate_embedding(message.clean_content)

            # Use similar memories from the past
            summaries = find_best_summaries(message.guild.id, embedding)
            summary = "\n\n".join(summaries)

        # Fetch info about the trigger
        if SUPER_SMART:
            who = await who_is(message.author.id, max_length=WHO_IS_CONTEXT_LENGTH)

        # Use the last few messages from the channel as context
        messages = con.execute(
            """
            SELECT content, names.name as username
            FROM messages
            LEFT JOIN names ON names.id=messages.author
            LEFT JOIN blocked_users ON messages.author=blocked_users.id
            WHERE channel=? AND blocked_users.id IS NULL
            ORDER BY date DESC 
            LIMIT ?
            """,
            (message.channel.id, MAX_HISTORY_N),
        ).fetchall()

        messages = [(m[0].replace("\n", " "), m[1]) for m in messages]
        messages = [f"{m[1]}: {m[0]}" for m in messages]

        messages = messages[0:FIXED_HISTORY_N] + drop_until(
            messages[FIXED_HISTORY_N:], HISTORY_LENGTH
        )

        messages.reverse()

        # Concatenate
        history = "\n".join(messages).replace("HagridBot", "Hagrid")

        # Define Hagrid
        system_prompts = "You are the loyal, friendly, and softhearted Rubeus Hagrid with a thick west country accent. You are a man of few words. If asked a question, you answers correctly. This is a conversation between one or more users, where you take part."

        # Build prompt
        # prompt = f"Context from the past:\n{summary}\n\n{message.author.name}'s profile summary:\n{who}\n\nConversation:\n{history}\n\nWhat short answer would Hagrid now respond? Respond with his thick west country accent!"
        prompt = f"Conversation:\n{history}\n\nWhat short answer would Hagrid now respond? Respond with his thick west country accent!"

        # Request
        await message.channel.send(
            (
                await generate_text(
                    prompt=prompt,
                    system_prompt=system_prompts,
                    max_tokens=150,
                    temperature=0.8,
                    frequency_penalty=0.1,
                )
            )
            .replace("Hagrid:", "")
            .strip()
        )

        return True
