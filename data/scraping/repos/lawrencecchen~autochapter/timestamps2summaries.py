#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path
from typing import List
import whisper
import os
import json
import cohere
import httpx
import asyncio
import cv2
import pytesseract
from PIL import Image

pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'



# In[3]:


# COHERE_API_KEY = "Dbvh24GpdsMeb3Gs4K87iy0zh3ruYN8znXDXBSqE"
# COHERE_API_KEY = "smw4Q4kVBOzzmsy5L6VHtQ0bjzc70rqEalAwNsWt"
COHERE_API_KEY = "2eXv0jtdPHV3tpe6ak9w5NmHDd1cV1n4mwLu8pLL"
cohere = cohere.Client(COHERE_API_KEY)


# In[4]:


# def timestamps2summaries(video_url: str, timestamps: List[int], output_dir: str):
#   transcribed = model.transcribe(video_url)
#   print(transcribed)
#   # transcribed["timestamps"]


# In[5]:


print("Loading Whisper")
model = whisper.load_model("tiny")
print("Loaded Whisper")

def transcribe_video(video_url: str, save_path: str):
  # video_transcribed_path = video_url
  # video_transcribed_path = os.path.join(base_dir,"transcribed", video_url.split("/")[-1].split(".")[0] + ".json")
  print(save_path)
  if os.path.exists(save_path):
    with open(save_path) as f:
      return json.load(f)
  else:
    result = model.transcribe(video_url)
    with open(save_path, "w") as f:
      json.dump(result, f)
    return result


# In[6]:


# transcribed = transcribe_video("videos/cs61a_lec1.mkv")


# In[7]:


from functools import wraps, partial

def summarize(passage: str):
  response = cohere.generate(
    model='large',
    # model='c33d4e37-1621-4156-95b0-3877544c2098-ft',
    prompt=f"""Your task is to summarize each passage. When given the choice, bias towards shorter descriptions. Always use a 3rd person, dispassioned tone. Do NOT use first or second person.

Begin.

Passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn’t the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to “the dusty section of the dictionary” to find its latest words.

TLDR: Wordle has not gotten more difficult to solve.
--
Passage: ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised $190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, and Plato Capital.

TLDR: ArtificialIvan has raised $190 million in Series C funding.
--
Passage: So today we're going to be doing a software engineering lecture. This is a somewhat of an experiment. So I'm going to give you some backscore and why it's happening. So interesting note that in 2003, Yish and earlier, 61A had two days a week, Monday, Wednesdays of technical topics. And then Friday was always something else. Now sometimes that was some social implications thing where we talked about the impact of computing. I mean, as we were before, I was like, you're age at the time. So they would talk about alternate topics, whatever it may be.

TLDR: Introduction
--
Passage: Now poverty is one of those things that is surprisingly hard to quantify which is the first real issue for governments that are trying to address this issue. Incomes are the most used metric, and almost every statistic you have likely heard on the issue will say something like these people live on less than 2 dollars a day, and for what it’s worth we have done exactly the same thing already in this video. But there are two problems with this, the first is that some people can be extremely comfortable with not much income. Some retirees would be a good example of this. They might own their own home fully paid off and have a nice pile of cash savings so they are very comfortable, but with interest rates as low as they are they might technically have an income below the internationally accepted poverty line.

TLDR: Measuring poverty
--
Passage: {passage}

TLDR:""",
    # prompt=f'Passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn’t the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to “the dusty section of the dictionary” to find its latest words.\n\nTLDR: Wordle has not gotten more difficult to solve.\n--\nPassage: ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised $190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, and Plato Capital.\n\nTLDR: ArtificialIvan has raised $190 million in Series C funding.\n--\nPassage: {passage}\n\nTLDR:',
    max_tokens=30,
    # max_tokens=50,
    temperature=0.8,
    k=0,
    p=1,
    frequency_penalty=0.2,
    presence_penalty=0,
    stop_sequences=["--"],
    return_likelihoods='NONE')
  # print('Prediction: {}'.format(response.generations[0].text))
  return response.generations[0].text

def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run 

@async_wrap
def async_summarize(passage: str):
  return summarize(passage).strip()


# In[8]:


# summarize(transcribed["text"][0:2000])


# In[10]:


async def get_async(url):
    async with httpx.AsyncClient() as client:
        return await client.get(url)


# In[11]:


# all units should be in seconds
def get_pre_summarized_segments(transcribed, timestamps: List[int], max_duration=300)->List[str]:
  pre_summarized_segments = []
  ts = 0
  # ts = timestamps[0]
  max_ts = timestamps[-1] + max_duration
  i_timestamp = 1
  i_segment = 0
  segments = transcribed["segments"]
  # i_segment = 0
  # start_segment = timestamps["start"]
  # end_segment = timestamps["end"]

  for i in range(1, len(timestamps) + 1):
    # curr_timestamp = timestamps[i - 1]
    if i == len(timestamps):
      next_timestamp = max_ts
    else:
      next_timestamp = timestamps[i]
    # print(next_timestamp)
    prev_segment = ""
    segment_text = ""

    while ts <= next_timestamp and i_segment < len(segments):
      seg = segments[i_segment]
      segment_text += seg["text"]
      ts = seg["start"] 
      i_segment += 1
    
    if segment_text == "":
      segment_text = prev_segment

    pre_summarized_segments.append(segment_text)
    prev_segment = segment_text
  return pre_summarized_segments
  #   # progress to current time
  #   while ts < curr_timestamp:
  #     segment = segments[i_segment]
  #     ts = segment["start"]
  #     i_segment += 1
    
  #   # get all segments until next timestamp
  #   while ts < next_timestamp:
  #     segment = segments[i_segment]
  #     segment_text += segment["text"]
  #     ts += segment["start"]
  #     i_segment += 1
    
  #   # add segment to pre_summarized_segments
  #   pre_summarized_segments.append(segment_text)
  
  # # get last segment
  # segment_text = ""
  # while ts < max_ts:
  #   segment = segments[i_segment]
  #   segment_text += segment["text"]
  #   ts += segment["start"]
  #   i_segment += 1
  # pre_summarized_segments.append(segment_text)

def extract_meta(video_url: str, video_name:str, timestamps: List[int])->List[str]:
  cap = cv2.VideoCapture(video_url)
  i_timestamp = 0
  screenshot_paths = []
  video_screenshot_dir = os.path.join(os.getcwd(), "screenshots", video_name)
  Path(video_screenshot_dir).mkdir(parents=True, exist_ok=True)

  while True:
    ret, frame = cap.read()
    if not ret:
      break
    curr_timestamp = timestamps[i_timestamp]
    screenshot_path = os.path.join(video_screenshot_dir, f"{curr_timestamp}.jpg")
    image_write_success = cv2.imwrite(screenshot_path, frame)
    if not image_write_success:
      print(f"error writing screenshot {screenshot_path}")
    img_rgb = Image.frombytes('RGB', frame.shape[:2], frame, 'raw', 'BGR', 0, 0)
    try:
      title = pytesseract.image_to_string(img_rgb, timeout=0.5)
      screenshot_paths.append({"screenshot_path":screenshot_path, "title": title})
    except RuntimeError as timeout_error:
      screenshot_paths.append({"screenshot_path":screenshot_path, "title": None})
    # print("extracting meta", i_timestamp, title)
    cap.set(cv2.CAP_PROP_POS_MSEC, curr_timestamp * 1000)
    i_timestamp += 1
    if i_timestamp >= len(timestamps):
      break
  cap.release()
  return screenshot_paths


async def summarize_segments(segments: List[str]):
  summaries = await asyncio.gather(*map(async_summarize, segments))
  return summaries

# segments = get_pre_summarized_segments(transcribed, fake_timestamps)
# summaries = await summarize_segments(segments)
# summaries
# summaries = [summarize(segment) for segment in segments]
# summaries


# In[13]:


# transcribed = transcribe_video("videos/cs61a_lec1.mkv")
# fake_timestamps = [0, 791]
# segments = get_pre_summarized_segments(transcribed, fake_timestamps)
# summaries = await summarize_segments(segments)

