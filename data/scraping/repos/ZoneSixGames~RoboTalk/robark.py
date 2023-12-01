# Bring in deps
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.memory import ConversationBufferMemory
import urllib.parse
from bark import generate_audio, SAMPLE_RATE, preload_models
import numpy as np
import nltk
from pydub import AudioSegment
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime

# Preload Bark models
preload_models()

# Application Framework
st.title('Zone Six Podcast Creator')
# Collect the inputs
prompt = st.text_input("Enter the podcast topic")
p1_name = st.text_input("Host Name", value='Libby')
p1 = st.text_input("Enter the personality for the Host", value='liberal Democrat woman, background in international affairs and world health, mirroring viewpoints of  nicole wallace, Stephen colbert, ali velshi, and christiane amanpour.')
p2_name = st.text_input("Enter second character's name:", value='Be Free')
p2 = st.text_input("Enter the personality for voice 2", value='weed smoking Libertarian bisexual woman from California who loves the environment, technology, and protecting individual freedoms from government overreach, mirroring the views of Spike Cohen, Elon Musk, Carl Sagan, and joe rogan.')
p3_name = st.text_input("Enter third character's name:", value='Q Anon 42')
p3 = st.text_input("Enter the personality for voice 3", value='right wing conservative Republican from Florida, follows government conspiracies on 4chan and reddit, mirroring the views of Milo Yiannopoulos, Ben Shapiro, Steve Bannon, and Tucker Carlson.')

p1_NAME = p1_name.upper()
p2_NAME = p2_name.upper()
p3_NAME = p3_name.upper()

# Map character names to voices
VOICE_MAP = {
    p1_name: "v2/en_speaker_1",  # host voice
    p2_name: "v2/en_speaker_2",  # guest1 voice
    p3_name: "v2/en_speaker_3",  # guest2 voice
}

PODCAST_DIR = None
# Load up the entries as environment variables
load_dotenv()

# Access the environment variables 
API_KEYS = {
    'OPENAI_API_KEY': st.text_input("Enter your OpenAI API key"),
    'GOOGLE_CSE_ID': st.text_input("Enter your Custom Search Engine ID (CSE) key"),
    'GOOGLE_API_KEY': st.text_input("Enter your Google API key"),
}

# Initialize environment
os.environ.update(API_KEYS)

# Initialize components
google_search_tool = GoogleSearchAPIWrapper()

# Initialize OpenAI API
openai_llm = OpenAI(model_name="gpt-3.5-turbo-16k") # Initialize the OpenAI LLM

# Define templates
title = PromptTemplate.from_template("Write a witty, funny, or ironic podcast title about {topic}.")
script = PromptTemplate.from_template("Write a podcast script based on a given title, research, and unique personalities. Title: {title}, Research: {news_research}, Personalities: Host: {p1_NAME}: {p1}, First Guest: {p2_NAME}: {p2}, Second Guest: {p3_NAME}: {p3}. The podcast should start with the Host giving an introduction and continue with the guest speakers as follows: {p1_NAME}: content n/ {p2_NAME}: Content n/ {p3_NAME}: content n/ and so on, replacing the host and guest names with the input names")
cont_script = PromptTemplate.from_template("Continue writing a podcast script based on a given title, research, recent podcast discussion history. Title: {title}, Research: {research}, Script: {script}")
news = PromptTemplate.from_template("Summarize this news story: {story}")

# Initialize chains
chains = {
    'title': LLMChain(llm=openai_llm, prompt=title, verbose=True, output_key='title'),
    'script': LLMChain(llm=openai_llm, prompt=script, verbose=True, output_key='script'),
    'cont_script': LLMChain(llm=openai_llm, prompt=cont_script, verbose=True, output_key='cont_script'),
    'news': LLMChain(llm=openai_llm, prompt=news, verbose=True, output_key='summary'),
}

# Initialize session state for script, research, title if they doesn't exist
if 'script' not in st.session_state:
    st.session_state.script = "Script will appear here"

if 'title' not in st.session_state:
    st.session_state.title = "Podcast Title Will Appear Here"
    
if 'cont_script' not in st.session_state:
    st.session_state.cont_script = ""
    
if 'news' not in st.session_state:
    st.session_state.news = ""
    
if 'research' not in st.session_state:
    st.session_state.research = ""
    
if 'podcast_dir' not in st.session_state:
    st.session_state.podcast_dir = ""

#Define the functions

def extract_news_text(url):
    nltk.download('punkt')
    #"""Extract the text of a news story given its URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')

    # Concatenate all paragraphs into a single string
    story_text = ' '.join([p.get_text() for p in paragraphs])

    # Tokenize the story_text
    tokens = nltk.word_tokenize(story_text)

    # Only keep the first 4000 tokens
    tokens = tokens[:4000]

    # Rejoin the tokens into a single string
    story_text = ' '.join(tokens)
    
    return story_text

def get_top_news_stories(topic, num_stories=5):
    """Get the top num_stories news stories on the given topic."""
    # URL encode the topic to ensure it's valid in a URL
    topic = urllib.parse.quote_plus(topic)
    # Get the feed from the Google News RSS
    feed = feedparser.parse(f'https://news.google.com/rss/search?q={topic}')

    # Return the top num_stories stories
    return feed.entries[:num_stories]

def summarize_news_stories(stories):
    """Summarize each news story using the OpenAI model."""
    summaries = []
    total_tokens = 0
    for story in stories:
        # Extract the URL from the story metadata
        url = story.get('link', '')
        if url:
            # Extract the news text
            story_text = extract_news_text(url)

            # Generate a summary
            summary = chains['news'].run(story_text)

            # Add summary to the list if it doesn't exceed the token limit
            summary_tokens = len(summary.split())  # rough token count
            if total_tokens + summary_tokens <= 10000:
                summaries.append(summary)
                total_tokens += summary_tokens
            else:
                break  # stop if we reach the token limit
    return summaries

def create_podcast_directory():
    global PODCAST_DIR
    now = datetime.now()  # get current date and time
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")  # format as a string
    podcast_dir = f"Podcast_{date_time}"  # prepend "Podcast_" to the string

    if not os.path.exists(podcast_dir):
        os.makedirs(podcast_dir)
        
    PODCAST_DIR = podcast_dir
    return PODCAST_DIR  # Add this line

def convert_comments_to_audio(comments):
    """Generate audio for each comment in the script."""
    audio_files = []
    silence = np.zeros(int(0.5*SAMPLE_RATE))
    for comment in comments:
        voice_id = VOICE_MAP[comment['role']]
        audio_array = generate_audio(comment['text'], history_prompt=voice_id)  # Use Bark's generate_audio
        audio_file = f"{st.session_state.podcast_dir}/{comment['role']}_{comment['order']}.mp3"  # Save in podcast directory
        audio_array.export(audio_file, format="mp3")  # Export as mp3
        audio_files.append(audio_file)
    return audio_files

def parse_script(script):
    comments = []
    lines = script.split('\n')
    for i, line in enumerate(lines):
        if ':' in line:
            role, content = line.split(':', 1)
            if role and content:
                role = role.strip().upper()  # capitalize role
                comments.append({'role': role, 'text': content.strip(), 'order': i})
    return comments

def validate_inputs(prompt, p1, p2, p3):
    return all([prompt, p1, p2, p3])

def combine_audio_files(audio_files):
    combined = AudioSegment.empty()
    for audio_file in sorted(audio_files):
        segment = AudioSegment.from_mp3(audio_file)
        combined += segment
    return combined

#Operational Structure
if st.button('Generate Script') and validate_inputs(prompt, p1, p2, p3):
    # Research and summarize top news stories
    stories = get_top_news_stories(prompt)
    news_summaries = summarize_news_stories(stories)
    st.session_state.research = ' '.join(news_summaries)  # Join the list of summaries into a single string

    # Generate title
    title_result = chains['title'].run(topic=prompt)
    st.session_state.title = title_result  # Saving title directly to session state.

    # Generate and display initial script
    script_result = chains['script'].run(
        title=st.session_state.title, 
        news_research=st.session_state.research, 
        p1_NAME=p1_NAME, 
        p2_NAME=p2_NAME, 
        p3_NAME=p3_NAME, 
        p1=p1, 
        p2=p2, 
        p3=p3
    )
    st.session_state.script = script_result

    # Save the script in the session state and to a text file
    st.session_state.podcast_dir = create_podcast_directory()
    with open(f"{st.session_state.podcast_dir}/podcast_script.txt", 'w') as f:
        f.write(st.session_state.script)
    st.success(f"Script saved in {st.session_state.podcast_dir}/podcast_script.txt")
    with open(f"{st.session_state.podcast_dir}/podcast_research.txt", 'w') as f:
        f.write(st.session_state.research)
    st.success(f"Research saved in {st.session_state.podcast_dir}/podcast_research.txt")
    
if st.button('Continue Script') and validate_inputs(prompt, p1, p2, p3):
    # Generate and display initial script
    script_result = chains['cont_script'].run(
        title=st.session_state.title, 
        research=st.session_state.research, 
        script=st.session_state.script
    )
    st.session_state.script += str(script_result)

    # Save the script in the session state and to a text file
    with open(f"{st.session_state.podcast_dir}/podcast_script.txt", 'w') as f:
        f.write(str(st.session_state.script))
    st.success(f"Script saved in {st.session_state.podcast_dir}/podcast_script.txt")


# Download script
st.download_button("Download Script", data='\n'.join(st.session_state.script), file_name='podcast_script.txt', mime='text/plain')

# Display script from session state
st.write(f'Title: {st.session_state.title}')
st.write(f'Script: \n{st.session_state.script}')
st.write(f'\n{st.session_state.cont_script}')

print(st.session_state.podcast_dir)

if st.button('Create Voices') and st.session_state.script:
    comments = parse_script('\n'.join(st.session_state.script))
    st.session_state['audio_files'] = convert_comments_to_audio(comments)
    for i, audio_file in enumerate(st.session_state['audio_files']):
        st.audio(f"{st.session_state.podcast_dir}/podcast.mp3", format='audio/mp3')
        
if st.button('Combine Audio') and st.session_state.script:
    combined_audio = combine_audio_files(st.session_state['audio_files'])

    combined_audio.export(f"{st.session_state.podcast_dir}/complete_podcast.mp3", format='mp3')
    
    st.audio(f"{st.session_state.podcast_dir}/complete_podcast.mp3", format='audio/mp3')

if st.button('Download Podcast') and os.path.exists(f"{st.session_state.podcast_dir}/complete_podcast.mp3"):
    with open(f"{st.session_state.podcast_dir}/complete_podcast.mp3", 'rb') as f:
        bytes = f.read()
    st.download_button("Download Podcast", data=bytes, file_name=f"{st.session_state.podcast_dir}/complete_podcast.mp3", mime='audio/mpeg')

with st.expander('News Summaries'):
    st.write(st.session_state.research)
    
with st.expander('Script'):
    st.write(st.session_state.title)
    st.write(st.session_state.script)
    st.write(st.session_state.cont_script)
