import os
import streamlit as st
import pytube
import openai
from youtube_search import YoutubeSearch
import streamlit.components.v1 as components

def download_youtube_video(url):
    try:
        youtube = pytube.YouTube(url)
        video = youtube.streams.get_highest_resolution()
       
        video.download()
        
        st.success('Video downloaded successfully!')
        # Rename the downloaded file to "video.mp4"
        default_filename = video.default_filename
        new_filename = "video.mp4"
        os.rename(default_filename, new_filename)
        

            
    except Exception as e:
        st.error(f'Error downloading video: {str(e)}')

def download_youtube_video2(url):
    try:
        youtube = pytube.YouTube(url)
        video = youtube.streams.get_highest_resolution()
       
        video.download()
        
        st.success('Video downloaded successfully!')
        # Rename the downloaded file to "video.mp4"
        default_filename = video.default_filename
        new_filename = "video2.mp4"
        os.rename(default_filename, new_filename)
        

            
    except Exception as e:
        st.error(f'Error downloading video: {str(e)}')

mode = st.radio("Please select mode", ("Student", "Normal"))

def main():
    st.title("GPTTUBE: Convert youtube videos into seo blog post in seconds")
   
    url = st.text_input("Enter the YouTube video URL:")
    if mode == "Normal":
        if st.button("Download"):
            if url:
                download_youtube_video(url)
            else:
                st.warning("Please enter the YouTube video URL.")
    if mode == "Student":
        if st.button("Download"):
            if url:
                download_youtube_video2(url)
            else:
                st.warning("Please enter the video url")
    ga_code = """
    <!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-M26G6BJJT0"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-M26G6BJJT0');
</script>
    """
    st.markdown(ga_code, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# Import the AssemblyAI module
import assemblyai as aai

# Your API token is already set here
aai.settings.api_key = "62654efc37ad4c17b521afd8413c9a63"

# Create a transcriber object.
transcriber = aai.Transcriber()

# If you have a local audio file, you can transcribe it using the code below.
# Make sure to replace the filename with the path to your local audio file.
if mode == "Normal":
    transcript = transcriber.transcribe("./video.mp4")

if mode == "Student":
    transcript = transcriber.transcribe("./video2.mp4")
# Alternatively, if you have a URL to an audio file, you can transcribe it with the following code.
# Uncomment the line below and replace the URL with the link to your audio file.
# transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/espn-bears.m4a")

# After the transcription is complete, the text is printed out to the console.
auth_token = st.secrets["auth_token"]
openai.api_key = auth_token
text = transcript.text

if mode == "Normal":
    final_ans = openai.Completion.create(
    prompt = "Convert this into an SEO blog post also make it fun to read and intuitive while being seo friendly give it a suitable title"+ text,
    engine = "text-davinci-003",
    max_tokens = 500
)

if mode == "Student":
    final_ans = openai.Completion.create(
    prompt = "Give questions based on the transcript of this video they should be IMPORTANT QUESTIONS ONLY AND NOT SIDETRACKED QUESTIONS also generate a study plan for this with insights"+ text,
    engine = "text-davinci-003",
    max_tokens = 500
)


main_ans  = final_ans["choices"][0]["text"]
st.write(main_ans)
if mode == "Normal":
    image_prompt  = openai.Completion.create(
    prompt = "Generate an image prompt for the following SEO blog post"+ main_ans,
    engine = "text-davinci-003",
    max_tokens = 500

)
    proompt = image_prompt["choices"][0]["text"]

    image = openai.Image.create(
    prompt = proompt,
    size = "256x256"
)

    image_url = image["data"][0]["url"]

    st.image(image_url)
    st.write(proompt)

