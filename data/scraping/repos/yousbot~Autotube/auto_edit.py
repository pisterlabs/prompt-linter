from contextlib import contextmanager,redirect_stderr,redirect_stdout
import time
from rake_nltk import Rake
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from termcolor import colored
import datetime as dt
from collections import OrderedDict
import requests
import os, random, sys, getopt, glob
import math
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sys, os, getopt
from termcolor import colored
from moviepy.editor import *
import glob
from os import devnull
from colored import fore, back, style
import openai

PEXELS_API_KEY = 'pexels_api_key'
openai.api_key = 'openai_api_key'

desktop_path = "your_desktop_path_here"
project_folder = desktop_path + "/AutoTube/"
local_videos_file = project_folder+"local_videos.txt"
local_library_folder = project_folder+"locallibrary/"
transcript_folder = project_folder+"transcript/"
downloads_folder = project_folder+"downloads/"
videos_folder = project_folder+"videos/"
transcript_file = transcript_folder+"transcript.srt"
shorts_file = project_folder+"shorts.txt"
shorts_folder = project_folder+"shorts/"
temp_folder = project_folder+"temp/"
audio_file = project_folder+"audio/speech.mp3"
rake = Rake()
lem = WordNetLemmatizer()
SINGULAR_UNINFLECTED = ['gas', 'asbestos', 'womens', 'childrens', 'sales', 'physics']
video_usage_temp_file = temp_folder+"video_usage.txt"

SINGULAR_SUFFIX = [
    ('people', 'person'),
    ('men', 'man'),
    ('wives', 'wife'),
    ('menus', 'menu'),
    ('us', 'us'),
    ('ss', 'ss'),
    ('is', 'is'),
    ("'s", "'s"),
    ('ies', 'y'),
    ('ies', 'y'),
    ('es', 'e'),
    ('s', '')
]

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

## Singularize Word
def singularize_word(word):
    for ending in SINGULAR_UNINFLECTED:
        if word.lower().endswith(ending):
            return word
    for suffix, singular_suffix in SINGULAR_SUFFIX:
        if word.endswith(suffix):
            return word[:-len(suffix)] + singular_suffix
    return word

## Extract Keywords from TEXT USING RAKE ALGORITHM
def get_keywords(text):
    
    ## auto
    
    r = Rake()
    print(colored(" [ Extracting keywords from text ... ]",'blue'))
    r.extract_keywords_from_text(text)
    print(colored(" [ Raking phrases ... ]",'blue'))
    keywords = r.get_ranked_phrases()
    keys = []
    print(colored(" [ Lemmatizing keywords ... ]",'blue'))
    for i in keywords:
        lemmatized = lem.lemmatize(i,"v")
        singular = singularize_word(lemmatized)
        keys.append(singular)
    print(colored(" [ Removing duplicates ... ]",'blue'))
    
    ## manual
    #keys = list(text.split(" "))
    
    keys = list(dict.fromkeys(keys[:5]))
    keys = " ".join(keys)
    keys = ' '.join(OrderedDict.fromkeys(keys.split()))
    return keys


## Extract keywords from text using OpenAI
'''
def get_keywords(text):
    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt="Give me three words that visualize best this sentence, seperate them by space in one line :" + text,
    temperature=0.4,
    max_tokens=50,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    resp = str(response.choices[0].text)
    print(colored(" [ Visualization keywords : "+resp+" ] ",'blue'))
    return resp
'''

## merge videos
def merge_videos(input_folder,output_file):
    
    print(colored(" [ Started merging files under directory "+input_folder+" ... ] ",'blue'))
    videos_to_merge = []
    list_of_files = filter( os.path.isfile,glob.glob(input_folder + '*') )
    list_of_files = sorted( list_of_files, key = os.path.getmtime, reverse=False)
    for vid in list_of_files:
        videos_to_merge.append(VideoFileClip(vid))
    print(colored(" [ Adding fade effect to clips ... ] ",'blue'))
    fade_duration = 0.5 # 1-second fade-in for each clip
    clips = [clip.crossfadein(fade_duration) for clip in videos_to_merge]
    clips = [clip.crossfadeout(fade_duration) for clip in videos_to_merge]
    final_clip = concatenate_videoclips(clips,method='compose')
    final_clip.write_videofile(output_file, temp_audiofile=temp_folder+'temp-audio.m4a', remove_temp=True, codec="libx264", audio_codec="aac")
    print(colored(" [ Videos have been merged successfully ! ] ",'blue'))
    print(colored(" [ Your merged video is exported to : "+output_file+" ] ",'blue'))

## Download a video
def download_video(file_name,url):
    response = requests.get(url, stream=True)
    with open(file_name, 'wb') as f:
        for chunk in response.iter_content(chunk_size = 1024 * 1024):
            if chunk:
                f.write(chunk)
    print(colored(" [ Downloaded successfully ! ] \n",'blue'))

## PEXELS
# Pexels API Key : 563492ad6f91700001000001e096bfdca4e24c1880760240a4ed44a7
def Pexels(search_query, sequence_id, sub_sequence_length, sub_sequence_num):
    PAGE = 1
    numvids = 1
    PER_PAGE = 25
    BASE_VIDEO_URL = 'https://api.pexels.com/videos'
    PARAMS = {'query' : search_query, 'orientation' : 'landscape', 'size':'medium', 'page':PAGE, 'per_page' : PER_PAGE }
    response = requests.get(f"{BASE_VIDEO_URL}/search", params=PARAMS, headers={'Authorization': PEXELS_API_KEY})
    videos = response.json()['videos']
    urls = []
    print(colored(" [ Search Query : ",'blue')+colored(search_query,'yellow')+colored(" ]",'blue'))
    print(colored(" [ Getting Links ... ] ",'blue'))
    for vid in videos:
        for vidFile in vid['video_files']:	 
            if vidFile['width'] == 1920 and vidFile['height'] == 1080 and vid['duration'] >= sub_sequence_length:
                urls.append(vidFile['link'])
    print(colored(" [ Number of videos returned : ",'blue')+colored(len(urls),'yellow')+colored(" ]",'blue'))
    print(colored(" [ Chosing "+str(sub_sequence_num)+" Random Videos ... ]",'blue'))

    ## Pick random videos from the list & Download them
    chosen_url = random.sample(set(urls), sub_sequence_num)
    sub_sequence_id=0
    for url in chosen_url:
        print(colored(" [ Video selected : ",'blue')+colored(url,'yellow')+colored("]",'blue'))
        print(colored(" [ Starting Downloads ... ]",'blue'))
        file_name = downloads_folder + str(sequence_id) +"_"+ str(sub_sequence_id) +'_video_online.mp4'
        download_video(file_name,url)
        sub_sequence_id += 1
    print(colored(" [ "+str(numvids)+" videos downloaded. ] ",'blue'))


def video_archive(sequence_id,keywords):
    sequence_videos = [filename for filename in os.listdir(downloads_folder) if filename.startswith(str(sequence_id)+"_")]
    i = 1
    with open(local_videos_file, 'r') as f:
        last_line = f.readlines()[-1]
        last_id_archive = int(last_line.split('.')[0])
        for seq_vid in sequence_videos:
            seq_vid_new_name = str(last_id_archive + i)+".mp4"
            tags = ','.join([str(elem) for elem in keywords])
            os.system("cp "+downloads_folder+seq_vid+" "+local_library_folder+seq_vid_new_name)
            os.system("echo \""+seq_vid_new_name+";"+tags+"\" >> "+local_videos_file)
            os.system("echo \""+seq_vid_new_name+"\" >> "+video_usage_temp_file)
            i += 1

## Search in Local Video Library for Videos tagged with our keywords, or download them online
def get_videos(keywords, sequence_id, sequence_length):
    local_videos = []
    sequence_existing_videos = filter( os.path.isfile,glob.glob(downloads_folder + str(sequence_id)+'_*') )
    with open(local_videos_file,"r") as f:
        for line in f:
            video_name = line.split(';')[0] 
            video_tags = line.split(';')[1].split(',') 
            video_tags = [s.replace('\n', '') for s in video_tags]
            set_tags = set(video_tags)
            set_keywords = set(keywords)
            judgement = False
            if set_tags & set_keywords:
                if os.system("grep -q ^"+ video_name + " " + video_usage_temp_file):
                    local_videos.append(video_name)

    for vid in sequence_existing_videos:
        os.system("rm -rf "+vid)
        print(colored(" [ Video "+vid+" deleted ! ]",'blue'))

    if sequence_length % 7 != 0:
        sub_sequence_num = math.trunc((sequence_length/7))+1
    else:
        sub_sequence_num = math.trunc((sequence_length/7))
    sub_sequence_length = sequence_length/sub_sequence_num

    if not local_videos:
        pexels_search_query = ' '.join([str(elem) for elem in keywords])
        print(colored(" [ Video not found in local library. ]",'blue'))
        print(colored(" [ Searching online ... ]",'blue'))
        print(colored(" [ Launching Pexels API ... ]",'blue'))

        ## Calculate number of Sub-Sequences included in a sequence
        Pexels(pexels_search_query,sequence_id, sub_sequence_length, sub_sequence_num)
        print(colored(" [ Archiving downloaded videos ... ]",'blue'))
        video_archive(sequence_id,keywords)
        print(colored(" [ Archiving finished. ]",'blue'))

    else:
        credit = sub_sequence_num
        for i in range(1,sub_sequence_num+1):
            print(" CREDIT : " + str(credit) + " i = " + str(i))
            print(colored(" [ Search Query : ",'blue')+colored(keywords,'yellow')+colored(" ]",'blue'))
            print(colored(" [ "+str(len(local_videos))+" videos found in Local Library. ]",'blue'))
            os.system("cp "+local_library_folder+local_videos[i-1]+" "+downloads_folder+str(sequence_id)+"_"+str(i)+"_video_local.mp4")
            os.system("echo \""+local_videos[i-1]+"\" >> "+video_usage_temp_file)
            print(colored(" [ Video copied from Local Library to Downloads Folder. ]",'blue'))
            credit -= 1

        if credit > 0:
            pexels_search_query = ' '.join([str(elem) for elem in keywords])
            print(colored(" [ Local library exhausted !! ]",'blue'))
            print(colored(" [ Searching online ... ]",'blue'))
            print(colored(" [ Launching Pexels API ... ]",'blue'))
            Pexels(pexels_search_query, sequence_id, sub_sequence_length, credit)
            print(colored(" [ Archiving downloaded videos ... ]",'blue'))
            video_archive(sequence_id,keywords)
            print(colored(" [ Archiving finished. ]",'blue'))
                
            
def change_video(sequence_id_list):
    ## Loop through all sequence_IDs[]
    for id in sequence_id_list: 
        # 1. Get keywords from Transcript for Specific Line
        file = open(transcript_file)
        transcript_content = file.readlines()
        sequence_length = int(transcript_content[int(id)+1].split('--')[1])
        sequence_text = transcript_content[int(id)+2]
        # 2. Call get_videos
        sequence_keywords = get_keywords(sequence_text).split()
        get_videos(sequence_keywords, int(id), sequence_length)
        print("\n")

def cut_footages(sequence_id,sequence_length):

    print(colored(" [ Cleaning shorts.txt ... ]",'blue'))
    os.system("rm -rf "+shorts_file)
    os.system("touch "+shorts_file)

    if sequence_length % 7 != 0:
        sub_sequence_num = math.trunc((sequence_length/7))+1
    else:
        sub_sequence_num = math.trunc((sequence_length/7))
    mean_sub_sequence_length = math.trunc(sequence_length / sub_sequence_num)

    print(colored(" [ Indexing videos to trim. ]",'blue'))
    list_of_files_to_cut = filter( os.path.isfile,glob.glob(downloads_folder + str(sequence_id)+'_*') )
    for filename in list_of_files_to_cut:
        os.system("echo \""+filename+",0,0,0,"+str(mean_sub_sequence_length)+" \" >> "+ shorts_file)

    print(colored(" [ Calling Shortify.py ... ]",'blue'))
    os.system("python3 "+project_folder+"shortify.py -e")


def highlight_keywords():

    file = open(transcript_file)  
    content = file.readlines()
    num_lines = sum(1 for line in open(transcript_file))
    i = 0
    while i < num_lines:
        sequence_id = content[i]
        sequence_start_time = content[i+1].split('-')[0]
        sequence_end_time = content[i+1].split('-')[1]
        sequence_text = content[i+2]
        with suppress_stdout_stderr():
            sequence_keywords = " ".join(get_keywords(sequence_text).split())

        sequence_sentence = sequence_text.split()
        print("")
        for word in sequence_sentence:
            print(word+" ",end='')
            
        
        if not sequence_keywords :
            for word in sequence_sentence:
                print(colored(word+" ",'yellow'),end='')
        else :
            for word in sequence_sentence:
                if word in sequence_keywords:
                    print(colored(word+" ",'red'),end='')
                else:
                    print(word+" ",end='')
        
        i += 4

print("\n")
argumentList = sys.argv[1:]
options = "dacks"
long_options = ["Download","Assemble","Change","Keywords","Silence"]
arguments, values = getopt.getopt(argumentList, options, long_options)
os.system("rm -rf "+downloads_folder+".DS_Store")

## Loop Through All Transcript Sequences
for currentArgument, currentValue in arguments:

    if currentArgument in ("-d", "--Download"):
        file = open(transcript_file)  
        content = file.readlines()
        num_lines = sum(1 for line in open(transcript_file))

        if len(os.listdir(downloads_folder)) != 0:
            latest_file = max(glob.glob(downloads_folder+'*'), key=os.path.getmtime)
            i = int(latest_file.split('/')[-1].split('_')[0])
        else:
            i = 0

        while i < num_lines - 1:
            print(" i : " + str(i) + " num_lines : " + str(num_lines))
            os.system("rm -rf "+downloads_folder+".DS_Store")
            print(colored(" [ Processing video number : ",'blue')+colored(str(int(i)/4+1).split('.')[0],'yellow')+colored(" ]",'blue'))
            sequence_length = int(content[i+1].split('--')[1])
            sequence_text = content[i+2]
            sequence_keywords = get_keywords(sequence_text).split()
            get_videos(sequence_keywords, i, sequence_length)      

            time.sleep(6)
            i=i+4

    elif currentArgument in ("-a","--Assemble"):
        # Assemble Final Video
        os.system("rm -rf "+shorts_file)
        os.system("rm -rf "+downloads_folder+".DS_Store")
        os.system("rm -rf "+downloads_folder+"*video.mp4")
        os.system("rm -rf "+shorts_folder+"*")
        os.system("touch "+shorts_file)

        ## Step 3 : TRIM all videos
        file = open(transcript_file)  
        content = file.readlines()
        num_lines = sum(1 for line in open(transcript_file))

        for i in range(0,num_lines,4):
            print(colored(" [ Cutting footages of sequence "+str(i)+" ]",'blue'))
            sequence_length = int(content[i+1].split('--')[1])
            cut_footages(i,sequence_length)
            #i += 4
        
        merge_videos(shorts_folder,videos_folder+"final_video_without_subtitles.mp4")
        now = dt.datetime.now()
        current = now.year+now.month+now.day+now.hour+now.minute+now.second

    elif currentArgument in ("-c", "--Change"):
        # change videos
        sequence_id_list = []
        print("\n")
        list_ids_from_input = input(colored("   Please enter the list of IDs to change ( Ex : 0,4,8,12 ) : ",'blue','on_grey'))
        sequence_id_list = list_ids_from_input.split(",")
        change_video(sequence_id_list)

    elif currentArgument in ("-k", "--Keywords"):
        highlight_keywords()
        print("\n")
            
    elif currentArgument in ("-s","--silence"):
        os.system(" unsilence "+audio_file+" "+temp_folder+"temp_audio.mp3 -ao -av 1 -sv 0")
        os.rename(temp_folder+"temp_audio.mp3",audio_file)

print("\n")
