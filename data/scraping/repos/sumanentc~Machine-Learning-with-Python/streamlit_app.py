import streamlit as st
import wikipedia
import tiktoken
import nltk
import openai
from PIL import Image
from elevenlabs import set_api_key
from elevenlabs import generate
import numpy as np

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

st.title('AI Generated Podcast')

img = Image.open('./GenerativeAI/Generate-Podcast/ai-podcast.jpg')
st.image(img)


def split_text(input_text):
    split_texts = sent_tokenize(input_text)
    return split_texts


def create_chunks(split_sents, max_token_len=2500):
    current_token_len = 0
    input_chunks = []
    current_chunk = ""
    for sents in split_sents:
        sent_token_len = len(enc.encode(sents))
        if (current_token_len + sent_token_len) > max_token_len:
            input_chunks.append(current_chunk)
            current_chunk = ""
            current_token_len = 0
        current_chunk = current_chunk + sents
        current_token_len = current_token_len + sent_token_len
    if current_chunk != "":
        input_chunks.append(current_chunk)
    return input_chunks


def get_instruct_prompt(topic):
    return f"""
    You are a {topic} enthusiast who is doing a research for a podcast. Your task is to extract relevant information from the Result delimited by triple quotes. 
    Please identify 2 interesting questions and answers which can be used for a podcast discussion.
    The identified discussions should be returned in thr following format.
    - Highlight 1 from the text
    - Highlight 2 from the text
    """


def get_conv_prompt(podcast_name, podcastFacts):
    podcastPrompt = f"""
    You are a writer creating the script for the another episode of a podcast {podcast_name} hosted by \"Tom\" and \"Jerry\".
    Use \"Tom\" as the person asking questions and \"Jerry\" as the person providing interesting insights to those questions.
    Always specify speaker name as  \"Tom\" or \"Jerry\" to identify who is speaking.
    Make the convesation casual and interesting.
    Extract relevant information for the podcast conversation from the Result delimited by triple quotes.
    Use the below format for the podcast conversation.
    1. Introduction about the topic and welcome everyone for another episode of the podcast {podcast_name}.
    2. Tom is the main host.
    2. Introduce both the speakers in brief.
    3. Then start the conversation.
    4. Start the conversation with some casual discussion like what they are doing right now at this moment.
    5. End the conversation with thank you speech to everyone.
    6. Do not use the word \"conversation\" response.
    """
    return podcastPrompt + f"Result: ```{podcastFacts}```"


def get_chat_outputs(requestMessages):
    chatOutputs = []
    for request in requestMessages:
        chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=[
                                                      {"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                  ]
                                                  )
        chatOutputs.append(chatOutput)
    return chatOutputs


def get_conv_output(requestMessage):
    finalOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                               messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                         {"role": "user", "content": requestMessage}
                                                         ],
                                               temperature=0.7
                                               )
    return finalOutput.choices[0].message.content


def get_podcast_facts(chatOutputs):
    podcastFacts = ""
    for chats in chatOutputs:
        podcastFacts = podcastFacts + chats.choices[0].message.content
    return podcastFacts


def createPodcast(podcastScript, speakerName1, speakerChoice1, speakerName2, speakerChoice2):
    genPodcast = []
    podcastLines = podcastScript.split('\n\n')
    podcastLineNumber = 0
    for line in podcastLines:
        if podcastLineNumber % 2 == 0:
            speakerChoice = speakerChoice1
            line = line.replace(speakerName1 + ":", '')
        else:
            speakerChoice = speakerChoice2
            line = line.replace(speakerName2 + ":", '')
        genVoice = generate(text=line, voice=speakerChoice, model="eleven_monolingual_v1")
        genPodcast.append(genVoice)
        podcastLineNumber += 1
    return genPodcast


input_text = st.text_input(label="Enter Wikipedia URL", value="https://en.wikipedia.org/wiki/Lionel_Messi")

if not input_text:
    st.error("Valid Wikipedia URL needs to be provided")
else:
    wikipedia_id = input_text.rsplit('/', 1)[-1]

if input_text:
    if not wikipedia_id:
        st.error("Valid Wikipedia URL needs to be provided")
    topic_name = st.text_input(label="Enter the Podcast Topic ", value="Sports")
    podcast_name = st.text_input(label="Enter Podcast Name", value="Sport 101")
    openai_key = st.text_input(label="Enter OpenAI Key", type="password",help="https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key")
    elevenlabs_key = st.text_input(label="Enter ElevenLabs API Key", type="password", help="https://docs.elevenlabs.io/api-reference/quick-start/authentication")

    if wikipedia_id:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        input = wikipedia.page(wikipedia_id, auto_suggest=False)
        wiki_input = input.content
        split_sents = split_text(wiki_input)
        input_chunks = create_chunks(split_sents, max_token_len=2000)

    if not openai_key:
        st.error("OpenAI key needs to be provided!")

    if not elevenlabs_key:
        st.error("ElevenLabs key needs to be provided!")

    if openai_key and elevenlabs_key:
        openai.api_key = openai_key
        instructPrompt = get_instruct_prompt(topic_name)
        requestMessages = []
        podcast_facts = None
        conv_content = None
        genPodcast = None
        for text in input_chunks:
            requestMessage = instructPrompt + f"Result: ```{text}```"
            requestMessages.append(requestMessage)
        try:
            with st.spinner('Generating podcast content ...'):
                chatOutputs = get_chat_outputs(requestMessages)
                podcast_facts = get_podcast_facts(chatOutputs)
            st.success('Content Generated!')
        except Exception as ex:
            st.error(f"Exception occurred while interacting with OpenAI {ex}")
        if podcast_facts:
            try:
                with st.spinner('Generating conversational content ...'):
                    conv_prompt = get_conv_prompt(podcast_name, podcast_facts)
                    conv_content = get_conv_output(conv_prompt)
                st.success('Conversation Generated!')
            except Exception as ex:
                st.error(f"Exception occurred while interacting with OpenAI {ex}")
            if conv_content:
                st.text_area(label="", value=conv_content, height=200)
                try:
                    if elevenlabs_key:
                        set_api_key(elevenlabs_key)
                        with st.spinner('Generating podcast audio ...'):
                            speakerName1 = "Tom"
                            speakerChoice1 = "Adam"
                            speakerName2 = "Jerry"
                            speakerChoice2 = "Domi"
                            genPodcast = createPodcast(conv_content, speakerName1, speakerChoice1, speakerName2, speakerChoice2)
                except Exception as ex:
                    st.error(f"Exception occurred while interacting with ElevelLabs {ex}")
                if genPodcast:
                    with open("genPodcast.mpeg", "wb") as f:
                        for pod in genPodcast:
                            f.write(pod)

                    audio_file = open('genPodcast.mpeg', 'rb')
                    audio_bytes = audio_file.read()

                    st.audio(audio_bytes, format='audio/ogg')
