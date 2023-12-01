import argparse
import contextlib
from datetime import datetime
from logging import Logger
import gspreader.gspreader as gspreader
from youtube import youtube_search
from lookups import *
import sys
import spotipy
import sys
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from rich import print
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from oauth2client import file, client, tools
import openai
from time import sleep
from rich import print

sys.path.insert(0, "C:\RC Dropbox\Rivers Cuomo\Apps\spotify")
sys.path.insert(0, "C:\RC Dropbox\Rivers Cuomo\Apps")

from catalog.scripts.genius_rc import *


"""
updates the data for existing rows in "Setlist", 
"Encyclopedia o' Riffs" using the spotify API, lyrics genius API, youtube API, and openai API
"""

print("\nsongdata.py")

debug = False

def cell_contains_data(cell: str):
    return cell not in ["", "--", "unknown"]


def clean_lyrics(data):
    """
    A one-time function
    """
    for row in data:

        row["lyrics"] = (
            row["lyrics"]
            .replace(";", "\n")
            .replace("/", "\n")
            .replace("[Matt:]", "")
            .replace("[Rivers:]", "")
            .replace("\r", "")
            .replace("\n \n", "\n")
            .replace(" \n ", "\n")
            .replace("\n\n", "\n")
            .replace("\n ", "\n")
            .replace("Embed", "")
            .strip()
        )

        row["lyrics"] = remove_bad_lines(row["lyrics"])


# spotify search method
def durationsSongpopularityArtistidTrackid(data):

    print("\nchecking for missing Trackid, durations, songPopularity, Artistid ....")

    for row in data:
        # if there is no track id in the sheet try looking the track up as a search query
        if row["track_id"] in ["", "unknown"]:
            # continue
            row = fetch_missing_track_id(row)

        with contextlib.suppress(Exception):
            # print(
            #     f"looking up durationsSongpopularityArtistidTrackid for {row['track_id']}"
            # )
            result = spotify.track(row["track_id"], market="US")
            row["artist_id"] = result["artists"][0]["id"]
            if row["duration"] in ["", 0, 1]:
                row["duration"] = int(result["duration_ms"] / 1000)
            row["song_popularity"] = result["popularity"]

            if row["album_title"] == "":
                row["album_title"] = (
                    result["album"]["name"]
                    .replace(" (Deluxe)", "")
                    .replace(" [Deluxe Edition]", "")
                    .replace(" - Deluxe Edition", "")
                    .replace(" (Original Motion Picture Soundtrack)", "")
                )

            if row["preview_url"] == "":
                row["preview_url"] = result["preview_url"]
                print("preview_url: ", row["preview_url"])
        if row["track_id_alternate"] != "":
            result2 = spotify.track(row["track_id_alternate"], market="US")
            row["song_popularity"] = max(result2["popularity"], result["popularity"])


def fetch_missing_track_id(row):
    search_str = str(row["artist_name"]) + ", " + str(row["song_title"])
    counter = 0
    while True:
        try:
            print(f"No track id in the sheet so Im querying Spotify for: {search_str}")
            result = spotify.search(search_str, type="track")
            # print(result)
            # exit()
            result = result["tracks"]["items"][counter]
            # print(result)
            # sleep(10000)

            # print(result['artists'][0]['name'])
            track_name = (
                str(result["name"])
                .lower()
                .split("-")[0]
                .replace("(remastered)", "")
                .strip()
            )
            # print(result['name'])
            # print(track_name)
            # str(row["song_title"]).lower()
            if (track_name == str(row["song_title"]).lower()) and (
                result["artists"][0]["name"].lower()
                == str(row["artist_name"].lower()).strip()
            ):
                row["track_id"] = result["id"]
                break

            counter = counter + 1
            # print(counter)
            if counter > 2:
                print(
                    f"{counter} cant find a track with the same song title and artist name on spotify for {search_str}. so leaving the track id cell empty"
                )
                result = None
                break
        except Exception:
            print("No match....or ", not_responding)
            sleep(0.5)
            Logger.log(row)
            counter = counter + 1
            if counter > 2:
                print(
                    f"{counter}cant find a track with the same song title and artist name on spotify for {search_str}. so leaving the track id cell empty"
                )
                row["track_id"] = "unknown"
                break
    return row


def ensure_headers(sheet, sheet_title):

    print("\nensuring headers...")
    existing_header_list = sheet.row_values(header_row_in_sheet)
    necessary_header_list = [
        "track_id",
        "artist_name",
        "song_title",
        "followers",
        "song_popularity",
        "artist_id",
        # "loudness",
        "tempo_spotify",
        # "time_signature_spotify",
        # "key_spotify",
        # "mode_spotify",
        "duration",
        "genres",
        "artist_popularity",
        "releaseDate",
        "lyrics",
        "youtube_views",
        # "danceability",
        # "energy",
        # "speechiness",
        # "acousticness",
        # "instrumentalness",
        # "liveness",
    ]
    col_count = sheet.col_count

    for necessary_header in necessary_header_list:

        if not necessary_header in existing_header_list:

            print("adding column for '{0}'".format(necessary_header))

            sheet.add_cols(1)
            sheet = sheets.open(sheet_title).get_worksheet(0)
            col_count = col_count + 1
            sheet.update_cell(1, col_count, necessary_header)


def fetch_openai(prompt,temperature):
    """ Fetches a reply from OpenAI's GPT-3 API"""

        
    base_prompt = """
    I want you to act as a show designer for Weezer's upcoming summer tour. The show is divided into four segments of contrasting moods. 
    1. The Pop Party section will feature upbeat, cheerful songs with a more light-hearted message about love and relationships. It will be high energy and encourage the audience to participate in singing along and dancing.
    2. The Emotional Ballads section will feature slower-paced songs with poignant lyrics about love and relationships. The instrumentals provide a reflective atmosphere that may be heavy at times, but ultimately brings comfort. 
    3. The Dark and Heavy section will focus on heavier topics such as mental health/anxiety struggles or social commentary/satire. These songs will have intense instrumentals to match the mood of the lyrical content, allowing for an honest exploration of these issues in a safe space.  
    4. the Fun and Uplifting segment will have a similar vibe but with deeper lyrics that speak to mental health/anxiety struggles or spiritual enlightenment. The instrumentals may be slightly slower paced than in the Pop Party section, however they are still uplifting as they celebrate hope for brighter days ahead.
    You are going to analyze 2 pieces of data: the song's spotify audio feature "energy" and the song's lyrics and then return 1 of these 4 strings, corresponding to the type of song you think it is. Make sure you only return one of these 4 strings with no other test around it. Here are the 4 strings in a list:
    ["1. Pop Party", "2. Emotional Ballads", "3. Dark and Heavy", "4 . Fun and Uplifting"]. 
    Now here are the song lyrics:
    """

    base_prompt = """
    I want you to act as a show designer for Weezer's upcoming summer tour. What are the themes of weezer?

    1. Love and Relationships 
    2. Growing Up 
    3. Social Commentary/Satire 
    4. Mental Health/Anxiety 
    5. Spirituality and Religion 
    6. Fun/Humor

    I want you to divide the show into four segments of contrasting moods. 

    1. Pop Party – Use bright, neon colors such as pink, blue, and green to create a lively atmosphere. Have dancers on stage and use props such as beach balls, pool floats, and umbrellas to really bring out the summer vibes. Add some special effects such as confetti and bubbles to really bring the energy up. 

    2. Emotional Ballads – Use subtle colors such as pastels and blues to create a more intimate atmosphere. Incorporate projections of summer memories, heartwarming images, and visuals of nature to enhance the emotion of the songs. 

    3. Dark and Heavy – Create a darker atmosphere with dimmed lights and colors such as black, red, and purple. Use visuals of darkness and despair to match the heaviness of the music. 

    4. Fun and Uplifting – Let the audience get lost in the music with bright lights and a variety of colors. Get creative with visuals and props such as fireworks, kites, and balloons to create a fun and uplifting mood that everyone can enjoy.


    Can you use some imagery and gags that are appropriate to the tour's main theme, indie rock Road Trip.

    1. Pop Party – Have a blow up car on stage that the band can hop into and out of while they are playing, as if they're taking an imaginary road trip during their set. 
    2. Emotional Ballads – Set up a vintage microphone stand with a map backdrop to give the vibe of an old-school roadside performance. 
    3. Dark and Heavy – Project images onto the stage that evoke feelings of loneliness or isolation like highway signs in desolate locations or empty roads at night time. 
    4. Fun and Uplifting – Get creative with props such as beach balls, colorful posters, inflatable guitars, sunglasses, etc., to give off a fun summer road trip vibe!

    1. Pop Party – Weezer sets off in their inflatable car to search for the legendary Lost Highway, full of excitement and anticipation. 
    2. Emotional Ballads – Although they are still determined to reach their destination, doubts start setting in as they come across obstacles that appear insurmountable. They find solace in each other's company and remember why it is important to keep going despite any setbacks along the way. 
    3. Dark and Heavy – As darkness falls on their journey, things become more uncertain but Weezer remembers why they started this quest in the first place - to find something better than what life has been throwing at them lately.
    4 . Fun and Uplifting– After some personal soul searching, Weezer finds strength within themselves and a renewed determination to press onward until they arrive at the island paradise waiting at the end of the Lost Highway!

    Could you elaborate on the difference in mood between the "pop party" and "fun and uplifting" sections? How are the songs different in each section?

    The Pop Party section will feature upbeat, cheerful songs with a more light-hearted message about love and relationships. It will be high energy and encourage the audience to participate in singing along and dancing. 

    Conversely, the Fun and Uplifting segment will have a similar vibe but with deeper lyrics that speak to mental health/anxiety struggles or spiritual enlightenment. The instrumentals may be slightly slower paced than in the Pop Party section, however they are still uplifting as they celebrate hope for brighter days ahead.

    In which section would you put these lyrics?
    """

    base_prompt = """
    I will give you a song's Energy rating - On a scale of 0 to 1, 1 being the most energetic. And I will give you the song's lyrics.

    """

    post_prompt = """Choose which category the song belongs to:
    1. Pop Party
    2. Emotional Ballads
    3. Dark and Heavy
    4. Fun and Uplifting
    """
    # post_prompt = 'Only return 1 of these 4 strings, corresponding to the type of song you think it is. Make sure you only return one of these 4 strings with no other text around it. Here are the 4 strings in a list: ["1. Pop Party", "2. Emotional Ballads", "3. Dark and Heavy", "4 . Fun and Uplifting"].'


    while True:
        try:
            reply = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=temperature, # higher is more creative, lower is more boring
                    max_tokens=2000, # speech tokens, not characters
                )
            break
        except Exception:
            print("OpenAI is not responding")
            sleep(10)

    reply = reply["choices"]
    return reply[0]["text"]


def get_args():
    if sys.argv[0] == "maintenance.py":
        sys.argv = ["songdata.py"]
    parser = argparse.ArgumentParser(
    description='updates the data for existing rows in "Setlist", "Encyclopedia o Riffs", even FOB'
)
    parser.add_argument(
    "-s", "--sheet", help="the sheet you want to get data for", required=False, choices=["setlist", "encyclopedia", "setlist.data", "all"], default="all"
)
    parser.add_argument(
    "-f", "--first", help="the first row you want to get data for", required=False
)
    parser.add_argument(
    "-m", "--method", help="the method you want to run", required=False, 
    choices=["show_sections", "clean_lyrics","lyrics", "tempoEnergyDance", "viewcount", "releaseDate", "genres_artist_names", "durationsSongpopularityArtistidTrackid","all"], 
    default="all"
)

    # # https://stackoverflow.com/questions/45078474/required-is-an-invalid-argument-for-positionals-in-python-command
    # parser.add_argument(
    #     "--songdata",
    #     help="Ignore. this is just for when you run maintenance.py with cmd line argument 'songdata'",
    #     required=False,
    # )
    # parser.add_argument('songdata', help='I just put it in here so maintenance will run the program without crashing when it foolishly passes "songdata" to songdata.py')
    # args = vars(parser.parse_args())
    args = parser.parse_args()
    return args


def get_show_section_from_openai(row):
    """ send the lyrics to openai and get one of 4 show sections: pop party, emotional ballads, etc. """
    lyrics = row["lyrics"]
    energy = row["energy"]
    prompt = base_prompt

    # if cell_contains_data(danceability):
    #     prompt = f"{prompt}\nDanceability: {danceability} out of 1.0"

    if cell_contains_data(energy):
        prompt = f"{prompt}\Here is the song's Spotify audio property 'energy' score: {energy} out of 1.0"

    if cell_contains_data(lyrics):
        prompt = f"{prompt}\Here are the song's lyrics: {lyrics}"

    prompt += post_prompt

    # print(prompt)

    reply = fetch_openai(prompt, 0.89).strip()
    print(row["song_title"],reply)
    return reply


def get_spotify():
    print("signing into spotify...")
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return spotify


def get_sheets():
    print("signing into google...")
    # get approved
    # scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
    scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.environ.get("ENCYCLOPEDIA_SERVICE_ACCOUNT"), scope)

    # print(creds)
    sheets = gspread.authorize(creds)
    return creds,sheets


# spotify artist method
def genres_artist_names(data):

    print(
        "\ngetting the missing permanent artist fields, Genres and ArtistNames. these search by artist_id... "
    )

    for row in data:
        artist_id = row["artist_id"]
        if not artist_id or artist_id == "unknown":
            continue

        if row["genres"] == "":
            while True:
                try:
                    # print(
                    #     '\nLooking up genres on Spotify for: {0}'.format(artist_id))
                    row["genres"] = str(spotify.artist(artist_id)["genres"])
                    break
                except:
                    print(not_responding)
                    sleep(1)

        if row["artist_name"] == "":
            while True:
                try:
                    # print(
                    #     '\nLooking up artist_name on Spotify for: {0}'.format(artist_id))
                    row["artist_name"] = spotify.artist(artist_id)["name"]
                    break
                except:
                    print(not_responding)
                    sleep(1)


def get_lyrics(data):

    print("\nupdating lyrics....")

    for row in data:

        if row["lyrics"] in ["", "!"]:
            print(
                "looking up lyrics for",
                row["artist_name"],
                row["song_title"],
                f"on genius",
            ),

            try:
                lyrics = lookupLyrics(row["artist_name"], row["song_title"], dense=True)

            except Exception as e:
                print(e)
                lyrics = "!"

            if len(lyrics) > 5000:
                lyrics = f'{row["artist_name"]}: {row["song_title"]}: Your input contains more than the maximum of 50000 characters in a single cell.'

            print(lyrics)

            row["lyrics"] = lyrics


def get_sheetlist(args):
    setlist_sheet = ("Setlist",0)
    setlist_data_sheet = ("Setlist","data")
    encyclopedia_sheet = ("Encyclopedia o' Riffs",0)
    sheetlist = [setlist_sheet, encyclopedia_sheet]
    if args.sheet:
            if args.sheet.lower() == "setlist":
                sheetlist = [setlist_sheet]
            elif args.sheet.lower() == "encyclopedia":
                sheetlist = [encyclopedia_sheet]
    return sheetlist


def releaseDate(data):
    """
    GET RELEASE_DATE METHOD TO FIND EARLIEST RELEASE_DATE  WITH SPOTIPY.SEARCH
    """

    print("\nchecking for earliest release_date with spotify search")
    # lookup missing durations from their search_str
    for row in data:

        new_release_date = "9999999999999999"
        new_release_date_2 = "9999999999999999"
        new_release_date_3 = "9999999999999999"
        new_release_date_4 = "9999999999999999"

        search_str = str(row["artist_name"]) + ", " + str(row["song_title"])
        artist = str(row["artist_name"]).lower().strip()
        name = str(row["song_title"]).lower().strip()

        # if row['releaseDate'] == '9999' or row['releaseDate'] == '':
        if row["releaseDate"] == "":
            # print(row['releaseDate'])
            result = search_spotify(search_str)

            with contextlib.suppress(Exception):
                resultName = (result["tracks"]["items"][0]["name"]).lower()
                resultArtist = (
                    result["tracks"]["items"][0]["artists"][0]["name"]
                ).lower()

                if resultArtist[:6] == artist[:6] and resultName[:4] == name[:4]:

                    new_release_date = result["tracks"]["items"][0]["album"][
                        "release_date"
                    ]

                    # print(resultArtist, resultName, 'match!', new_release_date)

                    if len(new_release_date) == 4:
                        new_release_date = f"{new_release_date}-01-01"
                        # print(new_release_date)
                else:
                    print(
                        " no match!",
                        resultArtist,
                        resultName,
                    )

            try:

                resultName = (result["tracks"]["items"][1]["name"]).lower()
                resultArtist = (
                    result["tracks"]["items"][0]["artists"][0]["name"]
                ).lower()
                # print(resultArtist, resultName)
                if resultArtist[:6] == artist[:6] and resultName[:4] == name[:4]:

                    new_release_date_2 = result["tracks"]["items"][1]["album"][
                        "release_date"
                    ]
                    # print(resultArtist, resultName,
                    #       'match!', new_release_date_2)
                    # print(result['tracks']['items'][1]['album']['artists'])
                    if len(new_release_date_2) == 4:
                        new_release_date_2 = new_release_date_2 + "-01-01"
                    # new_release_date = min(new_release_date, new_release_date_2)
                    # print(new_release_date)
                else:
                    print(
                        " no match!",
                        resultArtist,
                        resultName,
                    )

            except:
                pass
            try:

                resultName = (result["tracks"]["items"][2]["name"]).lower()
                resultArtist = (
                    result["tracks"]["items"][0]["artists"][0]["name"]
                ).lower()
                # print(resultArtist, resultName)
                if resultArtist[:6] == artist[:6] and resultName[:4] == name[:4]:
                    new_release_date_3 = result["tracks"]["items"][2]["album"][
                        "release_date"
                    ]
                    # print(resultArtist, resultName,
                    #       'match!', new_release_date_3)
                    # print(result['tracks']['items'][2]['album']['artists'])
                    if len(new_release_date_3) == 4:
                        new_release_date_3 = new_release_date_3 + "-01-01"
                    # new_release_date = min(new_release_date, new_release_date_3)
                    # print(new_release_date)
                else:
                    print(
                        " no match!",
                        resultArtist,
                        resultName,
                    )
            except:
                pass
            try:
                resultName = (result["tracks"]["items"][3]["name"]).lower()
                resultArtist = (
                    result["tracks"]["items"][0]["artists"][0]["name"]
                ).lower()
                # print(resultArtist, resultName)
                if resultArtist[:6] == artist[:6] and resultName[:4] == name[:4]:
                    new_release_date_4 = result["tracks"]["items"][3]["album"][
                        "release_date"
                    ]
                    # print('match!', new_release_date_4)
                    # print(result['tracks']['items'][3]['album']['artists'])
                    if len(new_release_date_4) == 4:
                        new_release_date_4 = new_release_date_4 + "-01-01"
                    # new_release_date = min(new_release_date, new_release_date_4)
                    # print(new_release_date)
                else:
                    print(
                        " no match!",
                        resultArtist,
                        resultName,
                    )
            except:
                pass

            # print(new_release_date)

            row["releaseDate"] = min(
                new_release_date,
                new_release_date_2,
                new_release_date_3,
                new_release_date_4,
            )[:4]
            # print(search_str, row['releaseDate'])

            # OK OFTEN 'REMASTERED' ETC CAUSES THE SONG TO NOT BE FOUND, RESULTING IN A RELEASE DATE OF 9999
            # why the hell would you do this?
            # if row['releaseDate'] == '9999':
            #     row['artist_name'] = firstresultArtist
            #     row['song_title'] = firstresultName

            """ YOU NEED TO FORMAT THE RELEASE DATE COLUMN AS PLAIN TEXT, NOT A DATE """


def search_spotify(search_str):
    while True:
        try:
            # print(
            #     '\nLooking up 4 release dates on Spotify for: {0}'.format(search_str))
            result = spotify.search(search_str)
            break
        except Exception:
            print(not_responding)
            sleep(1)
    return result


def show_sections(data):
    """ ask openai to choose a show section (of the setlist, pop, dark, ballad, fun) for each song """
    print("getting show sections...", end="")
    for row in data:

        if cell_contains_data(row["lyrics"]) and cell_contains_data(row["energy"]) and not cell_contains_data(row["show_section"]):
            show_section = get_show_section_from_openai(row)
            row["show_section"] = show_section
    print("done")
    return data

# spotify audio features method
def tempoEnergyDance(data):
    print("getting tempoEnergyDance from spotify...", end="")
    for row in data:
        if not cell_contains_data(row["danceability"]):
            track_id = row["track_id"]
            if not cell_contains_data(track_id):
                continue
            while True:
                try:
                    # print(
                    #     'Looking up audio_features on Spotify for: {0}'.format(track_id))
                    # ['loudness']                    #
                    result = spotify.audio_features(tracks=[track_id])[0]
                    # print(result)
                    break
                except:
                    print(not_responding)
                    sleep(1)
            # row["loudness"] = result["loudness"]
            row["danceability"] = result["danceability"]
            row["energy"] = result["energy"]
            # row["speechiness"] = result["speechiness"]
            # row["acousticness"] = result["acousticness"]
            # row["instrumentalness"] = result["instrumentalness"]
            # row["liveness"] = result["liveness"]
            row["tempo_spotify"] = result["tempo"]
    print("done")


def view_count(data):
    """
    viewCount. update views as a batch
    """
    print("view_count()")

    # get the data_sheet. This is different from the setlist sheet. It's the 'data' tab.
    data_sheet = gspreader.get_sheet("Setlist", "data")
    data_sheet_data = data_sheet.get_all_records()

    # Itereate through the setlist data (not the 'data' tab data)
    for row in data:

        if row["cover"].lower() == "x":
            row["youtube_views"] = ""
            continue

        song_title = str(row["song_title"])

        search_str = str(row["artist_name"]) + ", " + song_title

        if debug:
            # lookup youtube view count
            print(f"\nLooking up viewCount on youtube for: {search_str}: ")

        try:

            viewCount = youtube_views(search_str)

            # print(search_str, viewCount)

        except:

            # viewCount = "-"
            viewCount = ""

            # print(f'viewCount still missing for {search_str}')

        if debug:
            print(f"{song_title}: {viewCount:,}")

        data_row = next(
            (x for x in data_sheet_data if str(x["song_title"]).lower() == str(row["song_title"]).lower()), None
        )

        if data_row != None:
            data_row["youtube_views"] = viewCount
        # else:
        #     data_row["youtube_views"] = ""


    gspreader.update_range(data_sheet, data_sheet_data)

# Global Variables
not_responding = (
    "Spotify API not responding. Hold on for 5 seconds and we'll try again..."
)


args = get_args()
sheetlist = get_sheetlist(args)
header_row_in_sheet = 1
starting_row = int(args.first) - 2 if args.first else header_row_in_sheet -1
print("starting row in sheet", int(args.first) if args.first else "1")
print("starting_row of data", starting_row)
# method = args.method
print("method", args.method)

if args.method in ["durationsSongpopularityArtistidTrackid", "all", "tempoEnergyDance", "genres_artist_names" ]:

    spotify = get_spotify()

creds, sheets = get_sheets()

def main():
    

    for sheet_tuple in sheetlist:

        updated = False

        sheet = gspreader.get_sheet(sheet_tuple[0], sheet_tuple[1])

        sheet_title = ".".join(str(x) for x in sheet_tuple)

        print(f"\nExtracting all the data from the {sheet_title} sheet:")
        data = sheet.get_all_records()

        if args.method in ["clean_lyrics"]:
            clean_lyrics(data)
            updated = True

        if args.method in ["durationsSongpopularityArtistidTrackid", "all"]:
            durationsSongpopularityArtistidTrackid(data[starting_row:])
            updated = True

        if args.method in ["lyrics", "all"]:
            get_lyrics(data[starting_row:])
            updated = True

        if args.method in ["releaseDate", "all"]:
            releaseDate(data[starting_row:])
            updated = True

        if args.method in ["tempoEnergyDance", "all"]:
            tempoEnergyDance(data[starting_row:])
            updated = True

        if args.method in ["genres_artist_names", "all"]:
            genres_artist_names(data[starting_row:])
            updated = True

        if args.method in ["getlyrics","all"]: 
            get_lyrics(data[starting_row:])
            updated = True

        if sheet_title == "Setlist.0":

        #     if args.method in ["show_sections", "all"]:
        #         show_sections(data[starting_row:])
        #         updated = True

            # Just for the Setlist.data tab. doesn't change anything in Setlist.0
            if args.method in ["viewcount","all"]:  # or datetime.today().weekday() == 4:

                view_count(data[starting_row:])

        # # I HAVE TO RE-AUTHORIZE BECAUSE MY CREDS HAVE EXPIRED
        # # HOPEFULLY NOTHING HAS CHANGED IN THE HOUR SINCE YOU STARTED RUNNING THE PROGRAM???

        # sheets = gspread.authorize(creds)
        # sheet = sheets.open(sheet_title).get_worksheet(0)
        # print(data)

        if updated:

            gspreader.update_range(sheet, data)

    return "Success!"


if __name__ == "__main__":
    main()
