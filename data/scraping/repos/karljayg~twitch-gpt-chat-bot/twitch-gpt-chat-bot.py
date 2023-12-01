import irc.bot
import openai
import re
import tokensArray
import asyncio
import json
import random
import time
import urllib3
import threading
import signal
import sys
import requests
import logging
import math
import os
import spawningtool.parser
import wiki_utils
import tiktoken
import pygame
import pytz
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from settings import config
from datetime import datetime
from collections import defaultdict
from mathison_db import Database

# The contextHistory array is a list of tuples, where each tuple contains two elements: the message string and its
# corresponding token size. This allows us to keep track of both the message content and its size in the array. When
# a new message is added to the contextHistory array, its token size is determined using the nltk.word_tokenize()
# function. If the total number of tokens in the array exceeds the maxContextTokens threshold, the function starts
# deleting items from the end of the array until the total number of tokens is below the threshold. If the last item
# in the array has a token size less than or equal to the maxContextTokens threshold, the item is removed completely.
# However, if the last item has a token size greater than the threshold, the function removes tokens from the end of
# the message string until its token size is less than or equal to the threshold, and keeps the shortened message
# string in the array. If the total number of tokens in the array is still above the threshold after deleting the
# last item, the function repeats the process with the second-to-last item in the array, and continues deleting items
# until the total number of tokens is below the threshold. By using this logic, we can ensure that the contextHistory
# array always contains a maximum number of tokens specified by maxContextTokens, while keeping the most recent
# messages in the array.
global contextHistory
contextHistory = []


class GameInfo:
    def __init__(self, json_data):
        self.isReplay = json_data['isReplay']
        self.players = json_data['players']
        self.displayTime = json_data['displayTime']
        self.total_players = len(self.players)

    def get_player_names(self, result_filter=None):
        return [config.STREAMER_NICKNAME if player['name'] in config.SC2_PLAYER_ACCOUNTS else player['name'] for player
                in self.players if result_filter is None or player['result'] == result_filter]

    def get_player_race(self, player_name):
        lower_player_name = player_name.lower()
        for player in self.players:
            lower_name = player['name'].lower()
            if lower_name == lower_player_name:
                race = player['race'].lower()
                if race == 'terr':
                    return 'Terran'
                elif race == 'prot':
                    return 'Protoss'
                elif race == 'random':
                    return 'Rand'
                elif race == 'zerg':
                    return 'Zerg'
        return 'Unknown'  # Return a default value indicating the race is unknown
    
    def get_status(self):
        if all(player['result'] == 'Undecided' for player in self.players):
            return "REPLAY_STARTED" if self.isReplay else "MATCH_STARTED"
        elif any(player['result'] in ['Defeat', 'Victory', 'Tie'] for player in self.players):
            return "REPLAY_ENDED" if self.isReplay else "MATCH_ENDED"
        return None

    def get_winner(self):
        for player in self.players:
            if player['result'] == 'Victory':
                return player['name']
        return None


class LogOnceWithinIntervalFilter(logging.Filter):
    """Logs each unique message only once within a specified time interval if they are similar."""

    def __init__(self, similarity_threshold=0.95, interval_seconds=120):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.interval = timedelta(seconds=interval_seconds)
        self.last_logged_message = None
        self.last_logged_time = None
        self.loop_count = 0  # Initialize the loop counter
        self.loops_to_print = 5  # Number of loops to wait before printing

    # log filter for similar repetitive messages to suppress
    def filter(self, record):
        now = datetime.now()
        time_left = None
        self.loop_count += 1  # Increment the loop counter
        time_left = ...  # Calculate the time left
        time_since_last_logged = ...  # Calculate the time since last logged

        if self.last_logged_message:
            time_since_last_logged = now - self.last_logged_time
            time_left = self.interval - time_since_last_logged
            if time_since_last_logged < self.interval:
                similarity = SequenceMatcher(None, self.last_logged_message, record.msg).ratio()
                if similarity > self.similarity_threshold:
                    if self.loop_count % self.loops_to_print == 0:  # Check if it's time to print
                        print(f"suppressed: {math.floor(time_left.total_seconds())} secs")
                    return False

        self.last_logged_message = record.msg
        self.last_logged_time = now
        return True


# Initialize the logger at the beginning of the script
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addFilter(LogOnceWithinIntervalFilter())

# Set logging level for urllib3 to WARNING
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Player names of streamer to check results for
player_names = config.SC2_PLAYER_ACCOUNTS

def get_random_emote():
    emote_names = config.BOT_GREETING_EMOTES
    return f'{random.choice(emote_names)}'


# Global variable to save the path of the latest file found
latest_file_found = None


def find_latest_file(folder, file_extension):
    global latest_file_found

    try:
        if not os.path.isdir(folder):
            logger.debug(f"The provided path '{folder}' is not a directory. Please provide a valid directory path.")
            return None

        if not file_extension.startswith('.'):
            file_extension = '.' + file_extension

        logger.debug(f"Searching for files with extension '{file_extension}' in folder '{folder}' & subdirectories...")

        latest_file = None
        latest_timestamp = None

        for root, _, files in os.walk(folder):
            for filename in files:
                if filename.endswith(file_extension):
                    filepath = os.path.join(root, filename)
                    file_timestamp = os.path.getmtime(filepath)

                    if latest_file is None or file_timestamp > latest_timestamp:
                        latest_file = filepath
                        latest_timestamp = file_timestamp

        if latest_file:
            if latest_file == latest_file_found:
                logger.debug(f"The latest file with extension '{file_extension}' has not changed: {latest_file}")
            else:
                logger.debug(f"Found a new latest file with extension '{file_extension}': {latest_file}")
                latest_file_found = latest_file

            return latest_file
        else:
            logger.debug(
                f"No files with extension '{file_extension}' were found in the folder '{folder}' and its subdirectories.")
            return None

    except Exception as e:
        logger.debug(f"An error occurred while searching for the latest file: {e}")
        return None


class TwitchBot(irc.bot.SingleServerIRCBot):
    def __init__(self):

        self.first_run = True
        self.last_replay_file = None
        self.conversation_mode = "normal"
        self.total_seconds = 0
        self.encoding = tiktoken.get_encoding(config.TOKENIZER_ENCODING)
        self.encoding = tiktoken.encoding_for_model(config.ENGINE)

        # handle KeyboardInterrupt in a more graceful way by setting a flag when Ctrl-C is pressed and checking that
        # flag in threads that need to be terminated
        self.shutdown_flag = False
        signal.signal(signal.SIGINT, self.signal_handler)

        # threads to be terminated as soon as the main program finishes when set as daemon threads
        monitor_thread = threading.Thread(target=self.monitor_game)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Generate the current datetime timestamp in the format YYYYMMDD-HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Append the timestamp to the log file name
        log_file_name = config.LOG_FILE.replace(".log", f"_{timestamp}.log")
        # Set up the logging configuration
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Set bot configuration
        self.token = config.TOKEN
        self.channel = config.CHANNEL
        self.username = config.USERNAME
        self.server = config.HOST
        self.port = config.PORT
        self.ignore = config.IGNORE
        openai.api_key = config.OPENAI_API_KEY

        self.streamer_nickname = config.STREAMER_NICKNAME
        self.selected_moods = [config.MOOD_OPTIONS[i] for i in config.BOT_MOODS]
        self.selected_perspectives = [config.PERSPECTIVE_OPTIONS[i] for i in config.BOT_PERSPECTIVES]

        # Initialize the IRC bot
        irc.bot.SingleServerIRCBot.__init__(self, [(self.server, self.port, 'oauth:' + self.token)], self.username,
                                            self.username)
        # SC2 sounds
        with open(config.SOUNDS_CONFIG_FILE) as f:
            self.sounds_config = json.load(f) 

        # Initialize the database
        self.db = Database()

    def play_SC2_sound(self, game_event):
        if config.PLAYER_INTROS_ENABLED:
            if config.IGNORE_PREVIOUS_GAME_RESULTS_ON_FIRST_RUN and self.first_run:
                logger.debug ("per config, ignoring previous game on first run, so no sound will be played")
                return  
            try:
                # start defeat victory or tie is what is supported for now
                logger.debug(f"playing sound: {game_event} ")
                pygame.mixer.init()

                # Set the maximum volume (1.0 = max)
                pygame.mixer.music.set_volume(0.7)

                sound_file = random.choice(self.sounds_config['sounds'][game_event])
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play()
            except Exception as e:
                logger.debug(f"An error occurred while trying to play sound: {e}")
                return None
        else:
            logger.debug("SC2 player intros and other sounds are disabled")

    # incorrect IDE warning here, keep parameters at 3
    def signal_handler(self, signal, frame):
        self.shutdown_flag = True
        logger.debug(
            "================================================SHUTTING DOWN BOT========================================")
        self.die("Shutdown requested.")
        sys.exit(0)

    @staticmethod
    def check_SC2_game_status():
        if config.TEST_MODE_SC2_CLIENT_JSON:
            try:
                with open(config.GAME_RESULT_TEST_FILE, 'r') as file:
                    json_data = json.load(file)
                return GameInfo(json_data)
            except Exception as e:
                logger.debug(f"An error occurred while reading the test file: {e}")
                return None
        else:
            try:
                response = requests.get("http://localhost:6119/game")
                response.raise_for_status()
                return GameInfo(response.json())
            except Exception as e:
                logger.debug(f"Is SC2 on? error: {e}")
                return None

    def handle_SC2_game_results(self, previous_game, current_game):

        # do not proceed if no change
        if previous_game and current_game.get_status() == previous_game.get_status():
            # TODO: hide logger after testing
            #logger.debug("previous game status: " + str(previous_game.get_status()) + " current game status: " + str(current_game.get_status()))            
            return
        else:
            previous_game = current_game # do this here also, to ensure it does not get processed again
            if previous_game:
                pass
                #logger.debug("previous game status: " + str(previous_game.get_status()) + " current game status: " + str(current_game.get_status()))
            else:
                #logger.debug("previous game status: (assumed None) current game status: " + str(current_game.get_status()))                    
                pass
        response = ""
        replay_summary = ""  # Initialize summary string

        logger.debug("game status is " + current_game.get_status())

        # prevent the array brackets from being included
        game_player_names = ', '.join(current_game.get_player_names())

        winning_players = ', '.join(current_game.get_player_names(result_filter='Victory'))
        losing_players = ', '.join(current_game.get_player_names(result_filter='Defeat'))

        if current_game.get_status() in ("MATCH_ENDED", "REPLAY_ENDED"):

            result = find_latest_file(config.REPLAYS_FOLDER, config.REPLAYS_FILE_EXTENSION)
            # there are times when current replay file is not ready and it still finds the prev. one despite the SLEEP TIMEOUT of 7 secs
            # so we are going to do this also to prevent the bot from commenting on the same replay file as the last one
            if(self.last_replay_file == result):
                logger.debug("last replay file is same as current, skipping: \n" + result)
                return
            
            if result:
                logger.debug(f"The path to the latest file is: {result}")

                if config.USE_CONFIG_TEST_REPLAY_FILE:
                    result = config.REPLAY_TEST_FILE  # use the config test file instead of latest found dynamically

                # clear context history since replay analysis takes most of the tokens allowed
                contextHistory.clear()

                # capture error so it does not run another processSC2game
                try:
                    replay_data = spawningtool.parser.parse_replay(result)
                except Exception as e:
                    logger.error(f"An error occurred while trying to parse the replay: {e}")

                # Save the replay JSON to a file
                filename = config.LAST_REPLAY_JSON_FILE
                with open(filename, 'w') as file:
                    json.dump(replay_data, file, indent=4)
                    logger.debug('last replay file saved: ' + filename)

                # Players and Map
                players = [f"{player_data['name']}: {player_data['race']}" for player_data in
                           replay_data['players'].values()]
                region = replay_data['region']
                game_type = replay_data['game_type']
                unix_timestamp = replay_data['unix_timestamp']
                
                replay_summary += f"Players: {', '.join(players)}\n"
                replay_summary += f"Map: {replay_data['map']}\n"
                replay_summary += f"Region: {region}\n"
                replay_summary += f"Game Type: {game_type}\n"
                replay_summary += f"Timestamp: {unix_timestamp}\n"
                replay_summary += f"Winners: {winning_players}\n"
                replay_summary += f"Losers: {losing_players}\n"                

                # Game Duration
                frames = replay_data['frames']
                frames_per_second = replay_data['frames_per_second']

                self.total_seconds = frames / frames_per_second
                minutes = int(self.total_seconds // 60)
                seconds = int(self.total_seconds % 60)

                game_duration = f"{minutes}m {seconds}s"
                replay_summary += f"Game Duration: {game_duration}\n\n"
                logger.debug(f"Game Duration: {game_duration}")

                # Total Players greater than 2, usually gets the total token size to 6k, and max is 4k so we divide by 2 to be safe
                if current_game.total_players > 2: 
                    build_order_count = config.BUILD_ORDER_COUNT_TO_ANALYZE / 2
                else:
                    build_order_count = config.BUILD_ORDER_COUNT_TO_ANALYZE                    

                # Units Lost
                units_lost_summary = {player_key: player_data['unitsLost'] for player_key, player_data in
                                      replay_data['players'].items()}
                for player_key, units_lost in units_lost_summary.items():
                    player_info = f"Units Lost by {replay_data['players'][player_key]['name']}"  # ChatGPT gets confused if you use possessive 's vs by
                    replay_summary += player_info + '\n'
                    units_lost_aggregate = defaultdict(int)
                    if units_lost:  # Check if units_lost is not empty
                        for unit in units_lost:
                            name = unit.get('name', "N/A")
                            units_lost_aggregate[name] += 1
                        for unit_name, count in units_lost_aggregate.items():
                            unit_info = f"{unit_name}: {count}"
                            replay_summary += unit_info + '\n'
                    else:
                        replay_summary += "None \n"
                    replay_summary += '\n'

                # Build Orders
                build_orders = {player_key: player_data['buildOrder'] for player_key, player_data in
                                replay_data['players'].items()}

                # Separate players based on SC2_PLAYER_ACCOUNTS, start with opponent first
                player_order = []
                for player_key, player_data in replay_data['players'].items():
                    if player_data['name'] in config.SC2_PLAYER_ACCOUNTS:
                        player_order.append(player_key)
                    else:
                        player_order.insert(0, player_key)  # Put opponent at the start

                # Loop through build orders using the modified order
                for player_key in player_order:
                    build_order = build_orders[player_key]
                    player_info = f"{replay_data['players'][player_key]['name']}'s Build Order (first 20 steps):"
                    replay_summary += player_info + '\n'
                    for order in build_order[:int(build_order_count)]:
                        time = order['time']
                        name = order['name']
                        supply = order['supply']
                        order_info = f"Time: {time}, Name: {name}, Supply: {supply}"
                        replay_summary += order_info + '\n'
                    replay_summary += '\n'
                    
                # replace player names with streamer name
                for player_name in config.SC2_PLAYER_ACCOUNTS:
                    replay_summary = replay_summary.replace(player_name, config.STREAMER_NICKNAME)

                # Save the replay summary to a file
                filename = config.LAST_REPLAY_SUMMARY_FILE
                with open(filename, 'w') as file:
                    file.write(replay_summary)
                    logger.debug('last replay summary saved: ' + filename)

                # Save to the database
                try:
                    if self.db.insert_replay_info(replay_summary):
                        logger.debug("replay summary saved to database")
                    else:
                        logger.debug("replay summary not saved to database")
                except Exception as e:
                    logger.debug(f"error with database: {e}")

            else:
                logger.debug("No result found!")

        if current_game.get_status() == "MATCH_STARTED":
            # check to see if player exists in database
            try:
                #if game_type == "1v1":
                if current_game.total_players == 2:
                    logger.debug("1v1 game, so checking if player exists in database")
                    game_player_names = [name.strip() for name in game_player_names.split(',')]
                    for player_name in game_player_names:
                        logger.debug(f"looking for: {player_name}")
                        if player_name != config.STREAMER_NICKNAME:
                            result = self.db.check_player_exists(player_name, current_game.get_player_race(player_name))
                            if result is not None:

                                # Set the timezone for Eastern Time
                                eastern = pytz.timezone('US/Eastern')

                                # already in Eastern Time since it is using DB replay table Date_Played column
                                date_obj = eastern.localize(result['Date_Played'])

                                # Get the current datetime in Eastern Time
                                current_time_eastern = datetime.now(eastern)

                                # Calculate the difference
                                delta = current_time_eastern - date_obj

                                # Extract the number of days
                                days_ago = delta.days
                                hours_ago = delta.seconds // 3600
                                seconds_ago = delta.seconds

                                # Determine the appropriate message
                                if days_ago == 0:
                                    mins_ago = seconds_ago // 60
                                    if mins_ago > 60:
                                        how_long_ago = f"{hours_ago} hours ago."
                                    else:
                                        how_long_ago = f"{mins_ago} seconds ago."
                                else:
                                    how_long_ago = f"{days_ago} days ago"
                                
                                first_30_build_steps = self.db.extract_opponent_build_order(player_name)

                                msg = "Do both: \n"
                                msg += f"Mention all details here: {config.STREAMER_NICKNAME} played {player_name} {how_long_ago} in {{Map name}}," 
                                msg += f"and the result was a {{Win/Loss for {config.STREAMER_NICKNAME}}} in {{game duration}}. \n"
                                msg += f"As a StarCraft 2 expert, comment on last game summary. Be concise with only 2 sentences total of 25 words or less. \n"
                                msg += "-----\n"                                
                                msg += f"last game summary: \n {result['Replay_Summary']} \n"
                                self.processMessageForOpenAI(msg, "last_time_played")   

                                msg = f"Do both: \n"
                                msg += "First, print the build order exactly as shown. \n"
                                msg += "After, summarize the build order 7 words or less. \n"
                                msg += "-----\n"
                                msg += f"{first_30_build_steps} \n"                                                                
                                self.processMessageForOpenAI(msg, "last_time_played")   
                                
                            else:
                                msg = "Restate this without missing any details: \n "
                                msg += f"I think this is the first time {config.STREAMER_NICKNAME} is playing {player_name}, at least the {current_game.get_player_race(player_name)} of {player_name}"
                                logger.debug(msg)
                                self.processMessageForOpenAI(msg, "in_game")   
                            break  # avoid processingMessageForOpenAI again below
                        
            except Exception as e:
                logger.debug(f"error with find if player exists: {e}")

        elif current_game.get_status() == "MATCH_ENDED":
            if len(winning_players) == 0:
                response = f"Game with {game_player_names} ended with a Tie!"
                self.play_SC2_sound("tie")

            else:
                # Compare with the threshold
                if self.total_seconds < config.ABANDONED_GAME_THRESHOLD:
                    logger.debug("Game duration is less than " + str(config.ABANDONED_GAME_THRESHOLD) + " seconds.")
                    response = f"The game was abandoned immediately in just {self.total_seconds} seconds between {game_player_names} and so {winning_players} get the free win."
                    self.play_SC2_sound("abandoned")  
                else:
                    response = f"Game with {game_player_names} ended with {winning_players} beating {losing_players}"
                    if config.STREAMER_NICKNAME in winning_players:
                        self.play_SC2_sound("victory")
                    else:
                        self.play_SC2_sound("defeat")

        elif current_game.get_status() == "REPLAY_STARTED":
            self.play_SC2_sound("start")
            # clear context history so that the bot doesn't mix up results from previous games
            contextHistory.clear()
            response = f"{config.STREAMER_NICKNAME} is watching a replay of a game. The players are {game_player_names}"

        elif current_game.get_status() == "REPLAY_ENDED":
            winning_players = ', '.join(current_game.get_player_names(result_filter='Victory'))
            losing_players = ', '.join(current_game.get_player_names(result_filter='Defeat'))

            if len(winning_players) == 0:
                response = f"The game with {game_player_names} ended with a Tie!"
            else:

                # Compare with the threshold
                if self.total_seconds < config.ABANDONED_GAME_THRESHOLD:
                    response = f"This was an abandoned game where duration was just {self.total_seconds} seconds between {game_player_names} and so {winning_players} get the free win."
                    logger.debug(response)                    
                    self.play_SC2_sound("abandoned")  
                else:
                    if config.STREAMER_NICKNAME in winning_players:
                        self.play_SC2_sound("victory")
                    else:
                        self.play_SC2_sound("defeat")
                    response = (f"The game with {game_player_names} ended in a win for "
                                f"{winning_players} and a loss for {losing_players}")

        if not config.OPENAI_DISABLED:
            if self.first_run:
                logger.debug("this is the first run")
                self.first_run = False
                if config.IGNORE_PREVIOUS_GAME_RESULTS_ON_FIRST_RUN:
                    logger.debug("per config, ignoring previous game results on first run")
                    return  # exit function, do not proceed to comment on the result, and analysis on game summary
            else:
                logger.debug("this is not first run")

            # proceed
            self.processMessageForOpenAI(response, self.conversation_mode)

            # get analysis of game summary from the last real game's replay file that created, unless using config test replay file
            logger.debug("current game status: " + current_game.get_status() +
                        " isReplay: " + str(current_game.isReplay) +
                        " ANALYZE_REPLAYS_FOR_TEST: " + str(config.USE_CONFIG_TEST_REPLAY_FILE))

            # we do not want to analyze when the game (live or replay) is not in an ended state 
            # or if the duration is short (abandoned game)
            # unless we are testing with a replay file
            if ((current_game.get_status() not in ["MATCH_STARTED","REPLAY_STARTED"] and self.total_seconds >= config.ABANDONED_GAME_THRESHOLD)
                or (current_game.isReplay and config.USE_CONFIG_TEST_REPLAY_FILE)):
                # get analysis of ended games, or during testing of config test replay file
                logger.debug("analyzing, replay summary to AI: ")
                self.processMessageForOpenAI(replay_summary, "replay_analysis")
                # clear after analyzing and making a comment
                replay_summary = ""
            else:
                logger.debug("not analyzing replay")
                return

    def monitor_game(self):
        previous_game = None

        while True and not self.shutdown_flag:
            try:
                current_game = self.check_SC2_game_status()
                if(current_game.get_status() == "MATCH_STARTED" or current_game.get_status() == "REPLAY_STARTED"):
                    self.conversation_mode = "in_game"
                else:
                    self.conversation = "normal"
                if current_game:
                    if config.IGNORE_GAME_STATUS_WHILE_WATCHING_REPLAYS and current_game.isReplay:
                        pass
                    else:
                        time.sleep(2)  # wait so abandoned games doesnt result in false data of 0 seconds
                        self.handle_SC2_game_results(previous_game, current_game)

                previous_game = current_game
                time.sleep(config.MONITOR_GAME_SLEEP_SECONDS)
                # heartbeat indicator
                print(".", end="", flush=True)
            except Exception as e:
                pass

    # all msgs to channel are now logged
    def msgToChannel(self, message):
        self.connection.privmsg(self.channel, message)
        logger.debug("---------------------MSG TO CHANNEL----------------------")
        logger.debug(message)
        logger.debug("---------------------------------------------------------")

    def processMessageForOpenAI(self, msg, conversation_mode):

        # let's give these requests some breathing room
        time.sleep(config.MONITOR_GAME_SLEEP_SECONDS)

        # remove open sesame
        msg = msg.replace('open sesame', '')
        logger.debug(
            "----------------------------------------NEW MESSAGE FOR OPENAI-----------------------------------------")
        # logger.debug(msg)
        logger.debug('msg omitted in log, to see it, look in: "sent to OpenAI"')
        # remove open sesame
        msg = msg.replace('open sesame', '')

        # remove quotes
        msg = msg.replace('"', '')
        msg = msg.replace("'", '')

        # add line break to ensure separation
        msg = msg + "\n"

        # TODO: redo this logic
        # if bool(config.STOP_WORDS_FLAG):
        #    msg, removedWords = tokensArray.apply_stop_words_filter(msg)
        #    logger.debug("removed stop words: %s" , removedWords)

        # check tokensize
        total_tokens = tokensArray.num_tokens_from_string(msg, config.TOKENIZER_ENCODING)
        msg_length = len(msg)
        logger.debug(f"string length: {msg_length}, {total_tokens} tokens")

        # This approach calculates the token_ratio as the desired token limit divided by the actual total tokens. 
        # Then, it trims the message length based on this ratio, ensuring that the message fits within the desired token limit.
        # Additionally, the code adjusts the desired token limit by subtracting the buffer size before calculating the token ratio. 
        # This ensures that the trimming process takes the buffer into account and helps prevent the message from 
        # exceeding the desired token limit by an additional (BUFFER) of 200 tokens.

        # check tokensize
        total_tokens = tokensArray.num_tokens_from_string(msg, config.TOKENIZER_ENCODING)
        msg_length = len(msg)
        logger.debug(f"string length: {msg_length}, {total_tokens} tokens")
        if  int(total_tokens) > config.CONVERSATION_MAX_TOKENS:
            divided_by = math.ceil(len(msg) // config.CONVERSATION_MAX_TOKENS)
            logger.debug(f"msg is too long so we are truncating it 1/{divided_by} of its length")
            msg = msg[0:msg_length // divided_by]
            msg = msg + "\n" # add line break to ensure separation
            total_tokens = tokensArray.num_tokens_from_string(msg, config.TOKENIZER_ENCODING)
            msg_length = len(msg)
            logger.debug(f"new string length: {msg_length}, {total_tokens} tokens")

        # add User msg to conversation context if not replay nor last time played analysis
        if conversation_mode not in ["replay_analysis", "last_time_played"]:
            # add User msg to conversation context
            tokensArray.add_new_msg(contextHistory, 'User: ' + msg + "\n", logger)
            logger.debug ("adding msg to context history")
        else:
            contextHistory.clear()

        if conversation_mode == "last_time_played":
            # no mood / perspective
            pass
        else:

            # add complete array as msg to OpenAI
            msg = msg + tokensArray.get_printed_array("reversed", contextHistory)
            # Choose a random mood and perspective from the selected options
            mood = random.choice(self.selected_moods)

            if conversation_mode == "replay_analysis":
                perspective_indices = config.BOT_PERSPECTIVES[:config.PERSPECTIVE_INDEX_CUTOFF]  # say cutoff is 4, then select indices 0-3
            else:
                perspective_indices = config.BOT_PERSPECTIVES[config.PERSPECTIVE_INDEX_CUTOFF:]  # Select indices 4-onwards

            selected_perspectives = [config.PERSPECTIVE_OPTIONS[i] for i in perspective_indices]
            perspective = random.choice(selected_perspectives)

            if(conversation_mode == "normal"):
                # Add custom SC2 viewer perspective
                msg = (f"As a {mood} acquaintance of {config.STREAMER_NICKNAME}, {perspective}, "
                    + msg)
            else:
                if(conversation_mode == "in_game"):
                    msg = (f"As a {mood} observer of matches in StarCraft 2, {perspective}, comment on this statement: "
                        + msg)
                else:
                    msg = (f"As a {mood} observer of matches in StarCraft 2, {perspective}, "            
                        + msg)
        
        logger.debug("CONVERSATION MODE: " + conversation_mode)

        logger.debug("sent to OpenAI: %s", msg)
        completion = openai.ChatCompletion.create(
            model=config.ENGINE,
            messages=[
                {"role": "user", "content": msg}
            ]
        )
        try:
            if completion.choices[0].message is not None:
                logger.debug("completion.choices[0].message.content: " + completion.choices[0].message.content)
                response = completion.choices[0].message.content

                # add emote
                if random.choice([True, False]):
                    response = f'{response} {get_random_emote()}'

                logger.debug('raw response from OpenAI:')
                logger.debug(response)

                # Clean up response
                response = re.sub('[\r\n\t]', ' ', response)  # Remove carriage returns, newlines, and tabs
                response = re.sub('[^\x00-\x7F]+', '', response)  # Remove non-ASCII characters
                response = re.sub(' +', ' ', response)  # Remove extra spaces
                response = response.strip()  # Remove leading and trailing whitespace

                # dont make it too obvious its a bot
                response = response.replace("As an AI language model, ", "")
                response = response.replace("User: , ", "")
                response = response.replace("Observer: , ", "")
                response = response.replace("Player: , ", "")

                logger.debug("cleaned up message from OpenAI:")
                logger.debug(response)

                if len(response) >= 400:
                    logger.debug(f"Chunking response since it's {len(response)} characters long")

                    # Split the response into chunks of 400 characters without splitting words
                    chunks = []
                    temp_chunk = ''
                    for word in response.split():
                        if len(temp_chunk + ' ' + word) <= 400:
                            temp_chunk += ' ' + word if temp_chunk != '' else word
                        else:
                            chunks.append(temp_chunk)
                            temp_chunk = word
                    if temp_chunk:
                        chunks.append(temp_chunk)

                    # Send response chunks to chat
                    for chunk in chunks:
                        # Remove all occurrences of "AI: "
                        chunk = re.sub(r'\bAI: ', '', chunk)
                        self.msgToChannel(chunk)

                        # Add AI response to conversation context
                        tokensArray.add_new_msg(contextHistory, 'AI: ' + chunk + "\n", logger)

                        # Log relevant details
                        logger.debug(f'Sending openAI response chunk: {chunk}')
                        logger.debug(
                            f'Conversation in context so far: {tokensArray.get_printed_array("reversed", contextHistory)}')
                else:
                    response = re.sub(r'\bAI: ', '', response)
                    self.msgToChannel(response)

                    # Add AI response to conversation context
                    tokensArray.add_new_msg(contextHistory, 'AI: ' + response + "\n", logger)

                    # Log relevant details
                    logger.debug(f'AI msg to chat: {response}')
                    logger.debug(
                        f'Conversation in context so far: {tokensArray.get_printed_array("reversed", contextHistory)}')

            else:
                response = 'oops, I have no response to that'
                self.msgToChannel(response)
                logger.debug('Failed to send response: %s', response)
        except SystemExit as e:
            logger.error('Failed to send response: %s', e)

    def on_welcome(self, connection, event):
        # Join the channel and say a greeting
        connection.join(self.channel)
        logger.debug(
            "================================================STARTING BOT========================================")
        bot_mode = "BOT MODES \n"
        bot_mode += "TEST_MODE: " + str(config.TEST_MODE) + "\n"
        bot_mode += "TEST_MODE_SC2_CLIENT_JSON: " + str(config.TEST_MODE_SC2_CLIENT_JSON) + "\n"        
        bot_mode += "ANALYZE_REPLAYS_FOR_TEST: " + str(config.USE_CONFIG_TEST_REPLAY_FILE)  + "\n"
        bot_mode += "IGNORE_REPLAYS: " + str(config.IGNORE_GAME_STATUS_WHILE_WATCHING_REPLAYS) + "\n"
        bot_mode += "IGNORE_PREVIOUS_GAME_RESULTS_ON_FIRST_RUN: " + str(config.IGNORE_PREVIOUS_GAME_RESULTS_ON_FIRST_RUN) + "\n"
        bot_mode += "MONITOR_GAME_SLEEP_SECONDS: " + str(config.MONITOR_GAME_SLEEP_SECONDS) + "\n"
        logger.debug(bot_mode)                     

        prefix = ""  # if any
        greeting_message = f'{prefix} {get_random_emote()}'
        self.msgToChannel(greeting_message)

    def on_pubmsg(self, connection, event):

        # Get message from chat
        msg = event.arguments[0].lower()
        sender = event.source.split('!')[0]
        # tags = {kvpair["key"]: kvpair["value"] for kvpair in event.tags}
        # user = {"name": tags["display-name"], "id": tags["user-id"]}

        # Send response to direct msg or keyword which includes Mathison being mentioned
        if 'open sesame' in msg.lower() or any(sub.lower() == msg.lower() for sub in config.OPEN_SESAME_SUBSTITUTES):
            logger.debug("received open sesame: " + str(msg.lower()))
            self.processMessageForOpenAI(msg, self.conversation_mode)        
            return

        # search wikipedia
        if 'wiki' in msg.lower():
            logger.debug("received wiki command: /n" + msg)
            msg = wiki_utils.wikipedia_question(msg, self)
            logger.debug("wiki answer: /n" + msg)
            msg = msg[:500]  # temporarily limit to 500 char
            self.msgToChannel(msg)
            return

        # ignore certain users
        logger.debug("checking user: " + sender + " against ignore list")
        if sender.lower() in [user.lower() for user in config.IGNORE]:
            logger.debug("ignoring user: " + sender)
            return
        else:
            logger.debug("allowed user: " + sender)

        if config.PERSPECTIVE_DISABLED:
            logger.debug("google perspective config is disabled")
            toxicity_probability = 0
        else:
            toxicity_probability = tokensArray.get_toxicity_probability(msg, logger)
        # do not send toxic messages to openAI
        if toxicity_probability < config.TOXICITY_THRESHOLD:

            # any user greets via config keywords will be responded to
            if any(greeting in msg.lower() for greeting in config.GREETINGS_LIST_FROM_OTHERS):
                response = f"Hi {sender}!"
                response = f'{response} {get_random_emote()}'
                self.msgToChannel(response)
                # disable the return - sometimes it matches words so we want mathison to reply anyway
                # DO NOT return

            if 'bye' in msg.lower():
                response = f"bye {sender}!"
                self.msgToChannel(response)
                return

            if 'gg' in msg.lower():
                response = f"HSWP"
                self.msgToChannel(response)
                return

            if 'bracket' in msg.lower() or '!b' in msg.lower() or 'FSL' in msg.upper() or 'fsl' in msg.lower():
                response = f"here is some info {config.BRACKET}"
                self.msgToChannel(response)
                return

            # will only respond to a certain percentage of messages per config
            diceRoll = random.randint(0, 100) / 100
            logger.debug("rolled: " + str(diceRoll) + " settings: " + str(config.RESPONSE_PROBABILITY))
            if diceRoll >= config.RESPONSE_PROBABILITY:
                logger.debug("will not respond")
                return

            self.processMessageForOpenAI(msg, self.conversation_mode)        
        else:
            response = random.randint(1, 3)
            switcher = {
                1: f"{sender}, please refrain from sending toxic messages.",
                2: f"Woah {sender}! Strong language",
                3: f"Calm down {sender}. What's with the attitude?"
            }
            self.msgToChannel(switcher.get(response))


username = config.USERNAME
token = config.TOKEN  # get this from https://twitchapps.com/tmi/
channel = config.USERNAME


async def tasks_to_do():
    try:
        # Create an instance of the bot and start it
        bot = TwitchBot()
        await bot.start()
    except SystemExit as e:
        # Handle the SystemExit exception if needed, or pass to suppress it
        pass


async def main():
    tasks = [asyncio.create_task(tasks_to_do())]
    for task in tasks:
        await task  # Await the task here to handle exceptions

asyncio.run(main())
