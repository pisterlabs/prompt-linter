import streamlit as st
import openai
import requests
import os

# Load API keys from environment variables

OpenAI_Key = st.secrets["OpenAI_Key"]
yt_api_key = st.secrets["Youtube_Key"]

openai.api_key = OpenAI_Key
youtube_api_key = yt_api_key

def generate_recommendations(main_topic, subtopics):
    recommendations = []

    for subtopic in subtopics:
        prompt = f"Provide a step-by-step process for learning '{subtopic}' under the main topic '{main_topic}'."
        
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can try different engines
            prompt=prompt,
            max_tokens=200  # Adjust as needed
        )
        
        recommendations.append(response.choices[0].text.strip())

    return recommendations

def search_youtube_videos(query):
    base_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": youtube_api_key,
        "q": query,
        "part": "snippet",
        "type": "video",
        "maxResults": 1
    }
    response = requests.get(base_url, params=params).json()
    return response

def main():
    st.title("YouTube Video Recommendation App")

    main_topic = st.text_input("Enter the main topic:")
    subtopic1 = st.text_input("Enter subtopic 1:")
    subtopic2 = st.text_input("Enter subtopic 2:")

    if st.button("Generate Recommendations"):
        subtopics = [subtopic1, subtopic2]
        recommendations = generate_recommendations(main_topic, subtopics)
        
        for i, recommendation in enumerate(recommendations, start=1):
            # Search for a relevant video using the step-by-step process as the query
            query = f"{subtopics[i - 1]} tutorial"
            youtube_response = search_youtube_videos(query)
            
            if "items" in youtube_response:
                video_id = youtube_response["items"][0]["id"]["videoId"]
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                st.video(video_url)
                
            st.subheader(f"Step-by-Step Process for '{subtopics[i - 1]}':")
            st.write(recommendation)

if __name__ == "__main__":
    main()
