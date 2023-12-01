import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk
import googleapiclient.discovery
import openai

# Set the OpenAI API key
openai.api_key = "SET YOUR API KEY OVER HERE"

# Set the YouTube API key
youtube_api_key = "SET YOUR API KEY OVER HERE"

def get_popular_videos(query):
    # Build the YouTube API client
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=youtube_api_key)

    # Make a search request for videos on the given topic
    request = youtube.search().list(
        part="id,snippet",
        maxResults=5,  # Return up to 5 results
        q=query,      # Search for videos related to the given query
        type='video',  # Only include videos in the search results
        videoDefinition='high',  # Only include high-definition videos in the search results
        order='viewCount'  # Order the search results by view count, descending
    )

    # Execute the search request and parse the response
    response = request.execute()
    topics = "These are some popular YouTube videos on this topic:\n"
    for item in response['items']:
        video = {'title': item['snippet']['title']}
        topics += video['title'] + '\n'
    return topics

def chatgpt_conversation(conversation):
    # Call OpenAI's API to generate a response based on the conversation history
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',  # Use the GPT-3.5 Turbo model
        messages=conversation    # Pass the conversation history to the API
    )

    # Append the response to the conversation history and return the updated history
    conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return conversation

#function
def print_value():
    global value 
    value = entry.get()
    label.config(text="" + get_popular_videos(value) + 100 * "*") 
    conversation = []
    prompt = f"Please suggest three new video ideas on {value}.(max 600 character)"
    conversation.append({'role': 'user', 'content': prompt})
    conversation = chatgpt_conversation(conversation)
    label2.config(text="I'm thinking of new video ideas on this topic... Please wait...") 
    label3.config(text=" " + conversation[-1]['content'].strip(), font="calibri 7 bold") 

# Root window for the appearance
root = ttk.Window(themename="darkly")
root.title("TTInspire")
root.geometry("1200x600")

#area of entry for idea 
entry = tk.Entry(root)
entry.pack(padx=10, pady=10)

#Button
button = tk.Button(root, text="Find an idea and Trend Topics", command=print_value)
button.pack()

#first label, for the output like a print
label = tk.Label(root, text="")
label.pack()

#second label, for the output like a print
label2 = tk.Label(root, text="")
label2.pack()

#third label, for the output like a print
label3 = tk.Label(root, text="")
label3.pack()

root.mainloop()
