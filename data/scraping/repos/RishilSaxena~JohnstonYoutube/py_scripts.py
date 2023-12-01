import sys
import nltk
import string;
from youtube_transcript_api.formatters import TextFormatter
from gensim.models import Word2Vec
from gensim.models import Phrases
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup as bs
from openai import OpenAI
import requests
import re
import json
from youtube_transcript_api import YouTubeTranscriptApi
import os # To delete the temporary file
import csv # To read analytics data
import pandas as pd # To handle snippet selection
from pytube import YouTube # To downlaod the video
import moviepy.editor as mp # To crop videos


def getYoutubeTags(url):
    request = requests.get(url)
    html = bs(request.content, "html.parser")
    tags = html.find_all("meta", property="og:video:tag")
    result = [];
    for tag in tags:
        result.append(tag['content']);
    return result;


def scrape(query):
    url = f"https://www.googleapis.com/youtube/v3/search?key={sys.argv[3]}&q={query}&part=snippet,id&order=date&maxResults=20&type=video"

    r = requests.get(url)
    # print(r.status_code)

    data = r.json()
    # print(data)
    list_ids = [item['id']['videoId'] for item in data.get("items", [])]
    # print(list_ids)
    tagArray = [];
    for list_id in list_ids:
        tags = getYoutubeTags(f"https://www.youtube.com/watch?v={list_id}")
        for tag in tags:
            tagArray.append(tag.lower())

    # print(tagArray);
    tag_count = {}
    for element in tagArray:
        if element in tag_count:
            tag_count[element] += 1
        else:
            tag_count[element] = 1

    tag_count = sorted(tag_count.items(), key=lambda x:x[1], reverse=True)
    tag_dict = dict(tag_count)
    return tag_count

def remove_stopwords(sentences):
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word.casefold() not in stop_words]
        filtered_sentence = ' '.join(filtered_words)
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences

def remove_punctuation(sentences):
    punctuation = string.punctuation
    filtered_sentences = []

    for sentence in sentences:
        filtered_sentence = ''.join(char for char in sentence if char not in punctuation)
        filtered_sentences.append(filtered_sentence)

    return filtered_sentences

def generate_tags():
    transcript = YouTubeTranscriptApi.get_transcript(sys.argv[2])
    formatter = TextFormatter();
    text = formatter.format_transcript(transcript)

    text = re.sub(r'\[[0-9]*\]',' ',text) # [0-9]* --> Matches zero or more repetitions of any digit from 0 to 9
    text = text.lower() #everything to lowercase
    text = re.sub(r'\W^.?!',' ',text) # \W --> Matches any character which is not a word character except (.?!)
    text = re.sub(r'\d',' ',text) # \d --> Matches any decimal digit
    text = re.sub(r'\s+',' ',text) # \s --> Matches any characters that are considered whitespace (Ex: [\t\n\r\f\v].)

    sentences = sent_tokenize(text);
    sentences = remove_stopwords(sentences);
    word2vec_list = [];
    for sentence in sentences:
        word2vec_list.append(word_tokenize(remove_punctuation([sentence])[0]))

    bigram_transformer = Phrases(word2vec_list)
    model = Word2Vec(bigram_transformer[word2vec_list], min_count=4)
    most_common_words = model.wv.index_to_key[:20]
    most_common_words = [ele for ele in most_common_words if len(ele) >= 3]
    search_terms = ["mini itx build",
                        "fractal terra",
                        "fractal ridge",
                        "nzxt h1 v2 build",
                        ]
    for term in search_terms:
        other_tags = scrape(term)[:5]
        for tag in other_tags:
            most_common_words.append(tag[0])
    most_common_words = list(set(most_common_words))
    result = '['
    for word in most_common_words:
        result += ('"' + word + '",')
    result = result[0:len(result)-1]
    result += ("]")
    print(result)

# Generate Titles
def generate_titles_description(youtube_key, openai_key, video_id):
  client = OpenAI(api_key=openai_key)
  base_url = "https://www.googleapis.com/youtube/v3/videos"
  params = {
      "part": "snippet",
      "id": video_id,
      "key": youtube_key
  }
  try:
      response = requests.get(base_url, params=params)
      if response.status_code == 200:
          data = response.json()
          video = data["items"][0]
          description = video["snippet"]["description"]
      else:
          print("Request failed with status code:", response.status_code)
  except requests.exceptions.RequestException as e:
      print("An error occurred:", e)
      return e

  if (description):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": f"Use this video description to create a short title for this YouTube video: {description}"}
    ],
    max_tokens=20,
    n=3,
    temperature=0.4
    )

    titles_array = [choice.message.content.replace('"', '').replace("'", '') for choice in response.choices]
    return titles_array

def generate_titles_transcript(video_id, openai_key):
    client = OpenAI(api_key=openai_key)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_transcript = ""
    for segment in transcript:
        full_transcript += segment['text'] + " "
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": f"Generate a YouTube title based on the transcript of this YouTube video: \"{full_transcript}\""}],
      max_tokens=60,
      n=3
    )

    titles_array = [choice.message.content.replace('"', '').replace("'", '') for choice in response.choices]
    return titles_array

def generate_titles_queries(key, api_key, queries):
  client = OpenAI(api_key=api_key)
  titles = []

  for query in queries:
    api_url_query = f'https://www.googleapis.com/youtube/v3/search?key={key}&q={query}&type=video&order=viewCount&maxResults=20'

    try:
        r = requests.get(api_url_query)
        r.raise_for_status()
        data = r.json()

        ids = [item['id']['videoId'] if 'videoId' in item['id'] else item['id'] for item in data.get("items", [])]
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    for id in ids:
        api_url_video = f"https://www.googleapis.com/youtube/v3/videos?id={id}&key={key}&part=snippet"
        response = requests.get(api_url_video)

        if response.status_code == 200:
          video_data = response.json()
          titles.append(video_data['items'][0]['snippet']['title'])
        else:
          print("Error:", response.status_code)
          return response.status_code

  input_text = "\n".join(titles)

  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": f"Use these YouTube titles as inspiration to create a new one that would suit these queries: {', '.join(queries)}:\n{input_text}"}],
      max_tokens=20
  )

  return response.choices[0].message.content

def generate_with_numbers(youtube_key, openai_key, video_id):
  client = OpenAI(api_key=openai_key)
  base_url = "https://www.googleapis.com/youtube/v3/videos"
  params = {
      "part": "snippet",
      "id": video_id,
      "key": youtube_key
  }

  transcript = YouTubeTranscriptApi.get_transcript(video_id)
  full_transcript = ""

  for segment in transcript:
      full_transcript += segment['text'] + " "

  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": f"Using the provided transcript, generate a SHORT YouTube title with relevant numbers in it: {full_transcript}"}], # Context...
      max_tokens=20,
      n=2
  )

  titles_array = [choice.message.content.replace('"', '').replace("'", '') for choice in response.choices]
  return titles_array

def gen_titles_all(video_id, openai_key, youtube_key):
  first = generate_titles_description(youtube_key, openai_key, video_id)
  second = generate_titles_transcript(video_id, openai_key)
  third = generate_titles_queries(youtube_key, openai_key, first)
  fourth = generate_with_numbers(youtube_key, openai_key, video_id)

  res = [first, second, third, fourth]
  
  flattened_list = [item for sublist in res for item in (sublist if isinstance(sublist, list) else [sublist])]
  print(json.dumps(flattened_list))

def crop_with_retention(url: str, retention_path: str, output_path = "ConvertedShort", max_seconds=60, deadzones=None):   
    # os.remove("video.mp4") 
    # Gets important video metadata but does not download the mp4 file yet.
    video: YouTube = YouTube(url)
    seconds_per_percent = video.length/100

    with open(retention_path, "r") as file:
        csv_reader = csv.reader(file)
        # Get each moment's retention as a percent of the YouTube average at that time, skipping the header line.
        relative_retentions = [line[2] for line in csv_reader][1:]
    relative_retentions = [float(retention) for retention in relative_retentions]

    # The change in retention is more useful than the actual retention.
    retention_changes = [0]
    for i, retention in enumerate(relative_retentions[1:]):
        retention_changes.append(retention - relative_retentions[i])
    retention_changes = pd.Series(retention_changes)

    # Cut out deadzones that shouldn't be included in the final cut, such as advertisement reads.
    drops = []
    for deadzone in deadzones:
        lower_bound = int(deadzone[0] / seconds_per_percent)
        upper_bound = int(deadzone[1] / seconds_per_percent) + 1
        drops.extend(range(lower_bound, upper_bound+1))
    retention_changes.drop(drops, inplace=True)
    
    # Find the portions of the video with the best retention, getting as many as will fit into 1:00.
    max_snippets = int(max_seconds/seconds_per_percent)
    snippets = []
    num_snippets = 0
    while num_snippets < max_snippets:
        max = retention_changes.idxmax()
        # Convert percentage to actual timestamps
        snippets.append(max * seconds_per_percent)
        retention_changes.drop(max, inplace=True)
        num_snippets += 1
    snippets.sort()

    # Download the video. The file will be deleted after the function is done.

    stream_tag = video.streams.filter(file_extension="mp4")[0].itag
    stream = video.streams.get_by_itag(stream_tag)
    stream.download(filename="video_tmp.mp4")


    #buffer_neg = seconds_per_percent / 2
    #buffer_pos = seconds_per_percent - buffer_neg

    with mp.VideoFileClip("video_tmp.mp4") as to_clip:
        # Get a couple seconds after each top timestamp and combine these clips into a video.
        #clips = [to_clip.subclip(snippet-buffer_neg, snippet+buffer_pos) for snippet in snippets]
        clips = [to_clip.subclip(snippet, snippet+seconds_per_percent) for snippet in snippets]
        best_clips = mp.concatenate_videoclips(clips)

        # Crop the video to 16:9 vertical format instead of horizontal
        vid_width = best_clips.size[0]
        vid_height = best_clips.size[1]
        new_width = int(9/16 * vid_height)
        x_center = int(vid_width / 2)
        y_center = int(vid_height / 2)
        cropped_video = best_clips.crop(x_center=x_center, y_center=y_center, width=new_width, height=vid_height)
        cropped_video = cropped_video.without_audio()
        cropped_video.write_videofile(f"{output_path}.mp4")

    
    os.remove("video_tmp.mp4")
    print("complete")


match (sys.argv[1]):
    case 'generate_tags':
      generate_tags()
    case 'generate_titles':
      gen_titles_all(sys.argv[2], sys.argv[4], sys.argv[3])
    case 'shorten_video':
      crop_with_retention(url=sys.argv[2], retention_path="retention.csv", output_path='video', deadzones=[])