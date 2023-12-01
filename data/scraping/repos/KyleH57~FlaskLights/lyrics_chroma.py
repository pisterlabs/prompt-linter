import json
import os
import sqlite3
from collections import namedtuple
from json import JSONDecodeError

from syrics.api import Spotify

import openai

# takes a python dictionary?
def print_lyric_color_info(json_data):
    print("Primary color:", json_data['primaryColorRGB'])
    print("Accent color:", json_data['accentColorRGB'])
    print("Lyric associations:")
    for association in json_data['lyricAssociations']:
        print("    ", association['startTime'], "-", association['duration'], "ms", "-", association['colorRGB'], "-",
              association['reasoning'])



def parse_info(json_data):
    if type(json_data) == str:
        json_data = json.loads(json_data)
    json_data['primaryColorRGB'] = hex_to_rgb(json_data['primaryColorRGB'])
    json_data['accentColorRGB'] = hex_to_rgb(json_data['accentColorRGB'])
    for i in range(len(json_data['lyricAssociations'])):
        json_data['lyricAssociations'][i]['colorRGB'] = hex_to_rgb(json_data['lyricAssociations'][i]['colorRGB'])
    return json_data


def hex_to_rgb(color):
    if isinstance(color, str):
        if color.startswith('0x'):
            color = color[2:]  # Remove '0x' prefix
        elif color.startswith('#'):
            color = color[1:]  # Remove '#' prefix

        if len(color) != 6:
            raise ValueError("Invalid hexadecimal color code. It should be 6 characters long.")

        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    elif isinstance(color, list):
        if len(color) != 3:
            raise ValueError("Invalid RGB color list. It should contain exactly 3 values.")

        return tuple(color)

    else:
        raise TypeError("Invalid type for color. Expected str or list.")


def print_lyrics(unique_id):
    sp = Spotify(os.environ["SPOTIFY_COOKIE"])
    data = sp.get_lyrics(unique_id)
    # Print the lyrics line by line
    if 'lyrics' in data and 'lines' in data['lyrics']:
        for line in data['lyrics']['lines']:
            print(line['words'])


def print_raw_lyrics(unique_id):
    sp = Spotify(os.environ["SPOTIFY_COOKIE"])
    data = sp.get_lyrics(unique_id)
    print(data)


def make_gpt4_api_call(data, debug=False):
    """
    Makes a api call to GPT-4
    :param data: Lyrics string
    :param debug:
    :return:
    """
    if debug:
        print("Making GPT API call")
    openai.api_key = os.environ["OPENAI_API_KEY"]

    PROMPT = "Primary Color Association with a Song:\n\nRead through the lyrics of the song provided in the next " \
             "message and select a primary color that you believe encapsulates the overall mood, tone, and theme of " \
             "the song. Black, shades of grey, and silver are not to be used. Also, give an RGB color code for the " \
             "selected color.\n\nAdditionally, identify an accent color that complements your primary color choice" \
             " and further highlights the song's underlying themes or emotions. The accent color should also exclude" \
             " black, shades of grey, and silver. Provide the RGB color code.\n\n\nSpecific Lyric Color " \
             "Associations:\n\nIdentify specific lines in the song that you believe have a distinctive color " \
             "association, particularly in the context of common cultural understandings. For each line, select a" \
             " color that reflects its meaning or imagery, excluding black, shades of grey, and silver. Provide the " \
             "RGB color code for each selected color. The color associations " \
             "should be obvious and easy for someone to" \
             " understand while listening to the song. Try to do 6 per song, but if there are less than 6, that's ok." \
             " It is quality is more important than quantity. Do not do more than 6 per song. The associations should" \
             " be a times that are evenly distributed throughout the song. Do not use a line if the startTime is less" \
             " than 7000.\n\n\n\nOutput in json forat:\n{\n    \"primaryColorRGB\": \"<Primary color RGB code>\",\n" \
             "    \"accentColorRGB\": \"<Accent color RGB code>\",\n    \"lyricAssociations\": [\n        {\n  " \
             "          \"startTime\": \"<Start time of the line>\",\n           " \
             " \"colorRGB\": \"<Associated color RGB code>\",\n     " \
             "       \"reasoning\": \"<Only used in debug mode>\"\n        },\n        {\n        " \
             "    \"startTime\": \"<Start time of the line>\",\n          " \
             "  \"colorRGB\": \"<Associated color RGB code>\",\n         " \
             "   \"reasoning\": \"<Only used in debug mode>\"\n        },\n     " \
             "RR, GG, BB are from 0-FF. Do not provide any other data other than JSON data."

    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-16k",
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": str(data),
            }
        ],
        temperature=1.0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    if debug:
        print("GPT API call complete. Response:")

    if response['choices'][0]['finish_reason'] == 'length':
        print("API call failed. Response was too long.")
        return None
    elif response['choices'][0]['finish_reason'] == 'stop':
        return response
    else:
        print("API call failed. Reason: " + response['choices'][0]['finish_reason'])
        return None


def connect_to_db():
    """
    10 second timeout

    :return:
    """
    try:
        conn = sqlite3.connect('mydatabase.db', timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {e}")
        return None


def initialize_db(conn):
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mytable
                 (unique_id text PRIMARY KEY, data text, version text)''')
    conn.commit()


def fetch_song_from_db(conn, song_id):
    c = conn.cursor()
    c.execute('SELECT * FROM mytable WHERE unique_id=?', (song_id,))
    return c.fetchone()



def add_song_to_db(conn, song_id, debug=False):
    """
    This function is called when a song is not found in the database.
    It makes an API call to GPT-4 and saves the result to the database.
    :param conn:
    :param song_id:
    :param debug:
    :return:
    """
    try:


        sp = Spotify(os.environ["SPOTIFY_COOKIE"])
        lyric_data = sp.get_lyrics(song_id)
    except Exception as e:
        print("ERROR: Spotify API call failed.")
        print(e)
        return None

    if lyric_data is None:
        print("ERROR: Song may not have lyrics available.")
        return None

    response = make_gpt4_api_call(lyric_data, debug=debug)

    if response is None:
        print("ERROR: GPT response is None.")
        return None

    json_str = response['choices'][0]['message']['content']

    try:
        json_data = json.loads(json_str)


    except JSONDecodeError as e:
        print("ERROR: JSONDecodeError")
        print(e)
        print("json_str:")
        print(json_str)
        return None

    c = conn.cursor()


    add_duration_to_lyric_associations(json_data, lyric_data)

    if debug:
        print("GPT API Request complete:")
        print(json_data)

    json_data = parse_info(json_data)

    # Convert the Python dictionary back to a JSON string
    json_str_to_save = json.dumps(json_data)

    VERSION_NUMBER = "1.0"


    c.execute("INSERT OR REPLACE INTO mytable (unique_id, data, version) VALUES (?, ?, ?)",
              (song_id, json_str_to_save, VERSION_NUMBER))


    conn.commit()
    # print the database size

    # Count entries
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM mytable")
    count = cursor.fetchone()[0]
    print(f"Number of database entries: {count}")


    if debug:
        print("Data saved to database")
        print_lyric_color_info(json_data)





ColorDataStatus = namedtuple("ColorDataStatus", ["status", "data"])


def get_color_data2(song_id, replace=False, debug=False, fetch_only=True):
    """
    This is called when a song needs to be fetched from the database.
    If the song is not found in the database, it will be added.
    If the database is in use, it will wait up until the timeout.
    If the timeout is reached, it will return status "error".
    Returns success if song is in the database
    Returns not_found if song is not in the database and fetch_only is True
    :param song_id:
    :param replace:
    :param debug:
    :param fetch_only:
    :return:
    """
    if debug:
        print("Fetching color data for song ID:", song_id)
    # Initialize the database connection
    conn = connect_to_db()

    # Check if the database connection was successful
    if conn is None:
        return ColorDataStatus(status="error", data="Database connection error")

    # Create or validate database tables if they don't exist
    initialize_db(conn)

    # Fetch the song data from the database
    song_data = fetch_song_from_db(conn, song_id)

    # Case: Song found in the database and 'replace' flag is False
    if song_data and not replace:
        conn.close()  # Close the database connection
        return ColorDataStatus(status="success", data=json.loads(song_data[1]))

    # Case: Song found in the database and 'replace' flag is True
    elif song_data and replace:
        # If 'fetch_only' is False, update the song data in the database
        if not fetch_only:
            add_song_to_db(conn, song_id, debug=debug)
        conn.close()  # Close the database connection
        return ColorDataStatus(status="replaced", data="Song data replaced and updated")

    # Case: Song not found in the database
    else:
        # If 'fetch_only' is False, add the song data to the database
        if not fetch_only:
            try:
                add_song_to_db(conn, song_id, debug=debug)
            except ValueError as e:
                return ColorDataStatus(status="error", data=str(e))



        # Differentiate between "not found" and "added" based on 'fetch_only' flag
        if fetch_only:
            conn.close()  # Close the database connection
            # this is the case that happens on fa live song that is not in the database
            return ColorDataStatus(status="not_found", data="Song not found in database")
        else:
            song_data = fetch_song_from_db(conn, song_id)
            conn.close()  # Close the database connection
            if song_data is None: # some error occurred where there is a corrupt entry in the database
                # Corrupt entry found in database. Delete it.
                print("Corrupt entry found in database. Deleting it.")
                # open a connection and delete the entry
                conn = connect_to_db()
                c = conn.cursor()
                c.execute('DELETE FROM mytable WHERE unique_id=?', (song_id,))
                conn.commit()
                conn.close()
                return ColorDataStatus(status="error", data="Song not found in database")

            return ColorDataStatus(status="success", data=json.loads(song_data[1]))


def add_duration_to_lyric_associations(json_data, lyric_data):
    associations = json_data['lyricAssociations']
    # Access the 'lines' list from the 'lyrics' key in the dictionary
    lines = lyric_data.get('lyrics', {}).get('lines', [])

    # Extract the 'startTimeMs' values and convert them to integers
    lyric_times = [int(line['startTimeMs']) for line in lines if 'startTimeMs' in line]

    for association in associations:
        association_start = int(association['startTime'])
        next_lyric_time = next((time for time in lyric_times if time > association_start), None)

        if next_lyric_time is not None:
            association['duration'] = next_lyric_time - association_start
        else:
            association['duration'] = "N/A"  # or some default value
