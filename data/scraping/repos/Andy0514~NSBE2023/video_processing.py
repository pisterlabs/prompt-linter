from moviepy.editor import *
from os import listdir
from os.path import isfile, join, splitext
import os
import cohere
import pickle
import cosine_similarity
co = cohere.Client('yiOWD4KfXSiayGiim2MRmZRUvGsbdEFOY5QaCQ1Z') # This is your trial API key
import time

#  Setup - get the list of all available words in the dictionary
video_dict = dict()
video_dir = "videos"
for f in listdir(video_dir):
    if isfile(join(video_dir, f)):
        if (f[-3:] == "mp4"):
            video_dict[splitext(f)[0]] = join(video_dir, f)

print("Found " + str(len(video_dict)) + " videos")

def text_to_video(input):
    tempfile = "concatenated.mp4"
    if os.path.exists(tempfile):
        os.remove(tempfile)

    words = input.replace(".", "").lower().split()
    upperlim = len(words)
    i = 0
    clips = []

    while i < upperlim:
        w = words[i]

        # skipped words
        if (w == "be" or w == "of" or w == "is" or w == "are" or w == "a" or w == "an"):
            i += 1
            continue

        # special combination words
        if (i < upperlim - 1):
            if w == "i" and words[i+1] == "am":
                clips.append(VideoFileClip(join(video_dir, "i_am.mp4"), target_resolution=(240, 320)))
                i += 2
                continue

        if w in video_dict:
            clips.append(VideoFileClip(video_dict[w], target_resolution=(240, 320)))
        else:
            closest_word = cosine_similarity.find_closest_word(w)
            print("Found replacement: " + closest_word + " for " + w)
            clips.append(VideoFileClip(video_dict[closest_word], target_resolution=(240, 320)))
        i += 1

    finalClip = concatenate_videoclips(clips)
    finalClip.write_videofile(tempfile)
    return tempfile

def compute_embeddings_one_time_96(words):
    embeddings = dict()
    i = 0
    temp = []
    for k in words.keys():
        if (i < 95):
            temp.append(k.split()[0])
            i += 1
        else:
            temp.append(k.split()[0])
            response = co.embed(texts=temp, model="small")
            for j in range(len(temp)):
                embeddings[temp[j]] = response.embeddings[j]
                if (temp[j] == "decide"):
                    print(response.embeddings[j])
                    print(len(response.embeddings))
            i = 0
            temp = []
    return embeddings

def compute_embeddings_one_time(words):
    embeddings = dict()
    for k in words.keys():
        response = co.embed(texts=[k], model="small")
        embeddings[k] = response.embeddings[0]
        time.sleep(1)
        print(k)
        print(embeddings[k])
    return embeddings

'''
embed = compute_embeddings_one_time(video_dict)
with open('embeddings.pkl', 'wb') as fp:
    pickle.dump(embed, fp)
    print('Embeddings saved successfully to file')
'''

text_to_video("I have an apple")