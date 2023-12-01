# -*- coding: utf-8 -*-

# todo
#   The current name of the repo is "Financial Advisor" but it should be changed to something more general, thats just the
#       current example to get us started
#   I'll be opening issues for these and posting collaboration instructions soon, feel free at the assist in any way you
#       can, simply remember to create a new fork and follow common git practices for the time being. DO-NOT delete works
#       of others!!! instead contribute in modules and create a pull request.
#       Microservices (use of modules, classes, endpoints) is the way to go.
#       Most help appreciated for continuous integration and testing (CI/CD) and documentation.
#  - suggestions of better context for the question
#  - use pretrained models for a more robust solution both for the embeddings and the completions
#  - move from script to a web app
#  - add description of channel, etc... in the prompt
#  - add a way to add more context to the prompt in a more interactive way like a conversation
#  - sort the context by similarity and recentness, search web for more context
#  - relies on data preparation modules which should be chosen based on the data type itself
#     - Input Models
#       - youtube module
#       - "finBERT" engine_name for financial text classification and sentiment analysis of financial news
#       - Read financial reports module (10-K, 10-Q, 8-K, etc...) and extract context using NLP
#                                                                               (base models already exist)
#       - google speech to text module
#       - documentation reader module
#       - live streams module
#       - podcast module
#       - books module
#       - articles module
#       - research papers module
#     - Ensemble DB Schemes
#       - Prompt engineering
#       - Context choice
#       - Tokenizer module
#       - Context weighting for settings optimization and preparing any assisting data needed to answer the question
#       - Question Engineering
#     - Live Conversions
#       - Live explaining/translation of stream
#       - Live summarization of stream
#       - Live question answering of stream
#       - Live Actions based on stream
#   Not all these require either usage of embeddings or completion models. Modularity and speed are key,
#       although the latter is not a priority at the moment. Proof of concept and demo creation is the priority.
#       a priority at the moment. Proof of concept and demo creation is the priority.


#  - move transicrion downloaded episodes into a folder and not in the root directory

#  -
import pprint


import openai
import whisper
import pandas as pd
from pytube import YouTube
from getpass import getpass
from openai.embeddings_utils import get_embedding
from helper_functions import ask_question, is_part_of_question, combine_episodes

pd.set_option('display.max_columns', None)

pre_context_prompt = "Answer the following question using only the context below. Answer in the style of Ben Carlson a financial advisor and podcaster. If you don't know the answer for certain, say I don't know."

"""
Considerations when buying a new construction vs. an older home in an established neighborhood

e.g.
question = "Should I buy a house with cash?"
...
question = "Give me the best advice you have for someone who is just starting out in their career and has a nine week 
                beautiful baby girl as a 50 year old woman"
"""
question = "Blog about: how to be a millionaire by 30"

INPUT_DATA_FILE_NAME = 'example_questions_dat aset.csv'
TOP_N_CONTEXT = 2

N_EPISODES = -1
# COMPLETIONS_MODEL = "text-davinci-003"
# COMPLETIONS_MODEL = "davinci:ft-carbonyl-llc-2023-02-10-04-20-07"
# COMPLETIONS_MODEL = ""
EMBEDDINGS_MODEL = "text-embedding-ada-002"

TEMPERATURE = 0.5
MAX_TOKENS = 1500
MODEL_TOP_P = 1
FREQUENCY_PENALTY = 0.5
PRESENCE_PENALTY = 0.0


CONITNUE_TRAINING = True
SKIP_TRAINING = False
_SKIP_LOOP = False
ASK_QUESTION = True
#
SKIP_DOWNLOAD_AND_TRANSCRIBE = False
SKIP_EMBEDDINGS = False

if SKIP_EMBEDDINGS: SKIP_DOWNLOAD_AND_TRANSCRIBE = True  # ungainly

# Check which values for skip are valid and set the correct values
openai.api_key = getpass("Enter your OpenAI API Key")

df = pd.read_csv(INPUT_DATA_FILE_NAME)
print(f'{df = }')

# Set names for dirs and file locations
INPUT_DIR_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS = 'episodes_w_context_n_embedding'
PREFIX_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS = 'question_w_context_n_embedding'
OUTPUT_FILE_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS = 'questions_w_context_n_embedding.csv'
#
INPUT_DIR_FOR_EPISODES_WITH_CONTEXT = 'episodes_w_context'
PREFIX_FOR_EPISODES_WITH_CONTEXT = 'question_w_context'
OUTPUT_FILE_FOR_EPISODES_WITH_CONTEXT = 'questions_w_context.csv'
#
TEMP_DIR_FOR_TRANSCRIPTION = 'transcriptions'
TEMP_PREFIX_FOR_TRANSCRIPTION = 'transcription'


# Let's just get the questions for a single episode and make this work before we download and transcribe all episodes in bulk
def get_question_context(row):
    global transcription_output

    question_segments = list(
        filter(lambda segment: is_part_of_question(segment, row['start'], row['end']),
               transcription_output['segments']))
    # include question from timestamp in the context
    context = row['question']
    for segment in question_segments:
        context += segment['text']

    return context


# Read in all the episodes with context and embeddings and save to a single csv file for training the engine_name on

episodes_list = df['episode'].unique() if N_EPISODES > 0 else df['episode'].unique()[:N_EPISODES]

#create directories if they don't exist
import os
if not os.path.exists(INPUT_DIR_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS):
    os.makedirs(INPUT_DIR_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS)
if not os.path.exists(INPUT_DIR_FOR_EPISODES_WITH_CONTEXT):
    os.makedirs(INPUT_DIR_FOR_EPISODES_WITH_CONTEXT)
if not os.path.exists(TEMP_DIR_FOR_TRANSCRIPTION):
    os.makedirs(TEMP_DIR_FOR_TRANSCRIPTION)



if not SKIP_TRAINING:

    if CONITNUE_TRAINING:
        import glob
        import re

        # Get all files in the dir
        files = glob.glob(f'{INPUT_DIR_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS}/*')
        # Get all the episode numbers from the files
        episodes_with_context_n_embedding = [int(re.findall(r'\d+', file)[0]) for file in files]
        # Get the episodes that are not in the list
        episodes_list_to_add = list(set(episodes_list) - set(episodes_with_context_n_embedding))
        episodes_list_to_add.sort()

        # Get the last episode number
        last_episode = max(episodes_with_context_n_embedding)

        # Get the last file
        print(f'{last_episode = } {len(episodes_list) = }')

        if last_episode >= len(episodes_list):
            print(f'{episodes_list_to_add = }')
            print(f'{last_episode = } {len(episodes_list) = }')
            print("Looks like you have already processed all the episodes in input data.")
            SKIP_TRAINING = True
        else:
            index = episodes_list_to_add.index(last_episode)
            episodes_list = episodes_list[episodes_list_to_add.index(last_episode):]

    try:
        if not SKIP_TRAINING:
            for episode in episodes_list:
                if last_episode > len(episodes_list):
                    print(f'You have already processed all the episodes in input data.')
                    break
                else:
                    print(f'{episode = }')
                if not SKIP_DOWNLOAD_AND_TRANSCRIBE:
                    # Get rows for episode
                    episode_df = df[df['episode'] == episode].copy()
                    print(f'{episode_df = }')

                    # Download audio from YouTube for episode
                    # ['_age_restricted', '_author', '_embed_html', '_fmt_streams', '_initial_data', '_js', '_js_url', '_metadata',
                    # '_player_config_args', '_publish_date', '_title', '_vid_info', '_watch_html', 'age_restricted',
                    # 'allow_oauth_cache', 'author', 'bypass_age_gate', 'caption_tracks', 'captions', 'channel_id', 'channel_url',
                    # 'check_availability', 'description', 'embed_html', 'embed_url', 'fmt_streams', 'from_id', 'initial_data', 'js',
                    # 'js_url', 'keywords', 'length', 'metadata', 'publish_date', 'rating', 'register_on_complete_callback',
                    # 'register_on_progress_callback', 'stream_monostate', 'streaming_data', 'streams', 'thumbnail_url', 'title',
                    # 'use_oauth', 'vid_info', 'video_id', 'views', 'watch_html', 'watch_url']
                    youtube_video_url = episode_df['url'].iloc[0]  # assume all urls are the same for the episode
                    youtube_video = YouTube(youtube_video_url)
                    stream = youtube_video.streams.filter(only_audio=True).first()
                    stream.download(
                        filename=f'{TEMP_DIR_FOR_TRANSCRIPTION}/{TEMP_PREFIX_FOR_TRANSCRIPTION}_{episode}.mp4')
                    print(f'{youtube_video.description = }')

                    # Transcribe audio
                    print("Transcribing audio...")
                    model = whisper.load_model('base')
                    transcription_output = model.transcribe(
                        f'{TEMP_DIR_FOR_TRANSCRIPTION}/{TEMP_PREFIX_FOR_TRANSCRIPTION}_{episode}.mp4')
                    print(f"{transcription_output['text'] = }")

                    # Get context for each question
                    print("Getting question context...")
                    episode_df['context'] = episode_df.apply(get_question_context, axis=1)
                    episode_df.to_csv(
                        f'{INPUT_DIR_FOR_EPISODES_WITH_CONTEXT}/{PREFIX_FOR_EPISODES_WITH_CONTEXT}_{episode}.csv')
                else:
                    # Read in the episode with context
                    episode_df = pd.read_csv(
                        f'{INPUT_DIR_FOR_EPISODES_WITH_CONTEXT}/{PREFIX_FOR_EPISODES_WITH_CONTEXT}_{episode}.csv')

                if not SKIP_EMBEDDINGS:
                    # Get embeddings for each question
                    episode_df['embedding'] = episode_df['context'].apply(
                        lambda row: get_embedding(row, engine=EMBEDDINGS_MODEL))
                    episode_df.to_csv(
                        f'{INPUT_DIR_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS}/{PREFIX_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS}_{episode}.csv')
                else:
                    # Read in the episode with context and embeddings
                    episode_df = pd.read_csv(
                        f'{INPUT_DIR_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS}/{PREFIX_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS}_{episode}.csv')

            # combine all the episodes into a single csv file questions_w_context_n_embedding.csv
            combine_episodes(
                input_dir=INPUT_DIR_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS,
                prefix=PREFIX_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS,
                output_file=OUTPUT_FILE_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS)
        else:
            episode_df = pd.read_csv(OUTPUT_FILE_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS)

    except ValueError:
        # exit("Looks like you have already processed all the episodes in input data.")
        print("Looks like you have already processed all the episodes in input data.")
else:
    episode_df = pd.read_csv(OUTPUT_FILE_FOR_EPISODES_WITH_CONTEXT_AND_EMBEDDINGS)

if ASK_QUESTION:

    completion = ask_question(episode_df=episode_df, pre_context_prompt=pre_context_prompt, question=question,
                              top_n_context=TOP_N_CONTEXT,
                              completion_model=COMPLETIONS_MODEL,
                              embedding_model=EMBEDDINGS_MODEL,
                              temperature=TEMPERATURE,
                              max_tokens=MAX_TOKENS,
                              top_p=MODEL_TOP_P,
                              frequency_penalty=FREQUENCY_PENALTY,
                              presence_penalty=PRESENCE_PENALTY,
                              )

    print(f'{question = }')
    pprint.pprint(completion)
