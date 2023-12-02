# Import the necessary packages
import openai
import googleapiclient.discovery

# Set the OpenAI API key, Please refer to the readme file to learn how to obtain an API key.
openai.api_key = "You need to replace this part with your OpenAI API key"

# Set the YouTube API key, Please refer to the readme file to learn how to obtain an API key.
youtube_api_key = "You need to replace this part with your YouTube v3 API key"

# Define a function to get popular videos on a given topic using the YouTube Data API
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
    videos = "These are some popular YouTube videos on this topic:\n"
    for item in response['items']:
        video = {}
        video['title'] = item['snippet']['title']
        videos += video['title'] + '\n'
    return videos

# Prompt the user to enter a topic of interest and get the titles of popular videos on that topic
videos = get_popular_videos(input("What topic would you like to have new video ideas on?\n"))
print(videos)
print("I'm thinking of new video ideas on this topic... Please wait...\n")

# Define a function to generate three new video ideas on a given topic using OpenAI's GPT-3.5 language model
def ChatGPT_conversation(conversation):
    # Call OpenAI's API to generate a response based on the conversation history
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',  # Use the GPT-3.5 Turbo model
        messages=conversation    # Pass the conversation history to the API
    )

    # Append the response to the conversation history and return the updated history
    conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return conversation

# Prompt the user to suggest three new video ideas on the topic and generate responses using the ChatGPT_conversation function
conversation = []
prompt = f"Please suggest three new video ideas on {videos}."
conversation.append({'role': 'user', 'content': prompt})
conversation = ChatGPT_conversation(conversation)

# Print the response from the OpenAI model
print("Here they are! Some video ideas on similar topics.\n", conversation[-1]['content'].strip())
