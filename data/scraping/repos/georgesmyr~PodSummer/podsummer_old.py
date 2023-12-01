# OS libraries
import os
from pathlib import Path
import json

# Feed parser libraries
import feedparser

# Audio transcription libraries
import whisper

# LLM libraries
import openai
import tiktoken

# Guest information libraries
import wikipedia

# My utility libraries
import utils


class PodSummer:
    """

    """

    def __init__(self):
        """
        Initialisation of PodSummer instance
        """
        # OpenAI GPT Model
        self.chat_model = "gpt-3.5-turbo"
        self.setOpenAI_API_KEY()

        # Audio Transcription Model
        self.trans_model_path = "model/medium.pt"
        self.trans_model = None

        # Podcast Information
        self.podcast_title = None
        self.episode_number = None
        self.episode_title = None
        self.episode_subtitle = None
        self.episode_summary = None

        # Directories
        self.content_dir = Path('content/')
        self.content_dir.mkdir(exist_ok=True)
        self.podcast_dir = None
        self.episode_dir = None

        # Filepaths
        self.audio_path = None
        self.transcript_path = None
        self.summary_path = None
        self.highlights_path = None

    def setChatModel(self, model):
        """ Sets the chat model """
        self.chat_model = model

    def setOpenAI_API_KEY(self):
        """ Sets OpenAI API key """
        openai.api_key = utils.load_text('api_key.txt')

    def loadWhisper(self):
        """ Loads the whisper model """
        folder_path = os.path.dirname(self.trans_model_path)
        # Check if the file path exists
        if not os.path.exists(self.trans_model_path):
            # Check if the folder path exists, and if not, create the folder
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"The folder '{folder_path}' has been created.")

            print("Downloading Whisper model.")
            # noinspection PyProtectedMember
            whisper._download(whisper._MODELS["medium"], "model/", False)
            print("Download completed.")
        else:
            print(f"Whisper model found.")

        self.trans_model = whisper.load_model("medium")
        print("Model loaded")
        print("Device: ", self.trans_model.device)

    def extractEpisodeInfo(self, pod_feed, num):
        """
        Gets podcast feed and episode number starting from the end and extracts:
        1. Episode title
        2. Episode number
        3. Episode Summary (provided by the podcasters)
        if they exist.
        """
        episode = pod_feed.entries[num]
        self.episode_title = episode.title
        self.episode_number = len(pod_feed.entries) - num
        self.episode_summary = episode.summary

    def getPodcast(self, rss_url, num=0):
        """
        Downloads the audio from the last podcast in the rss_url
        and saves it in the local path
        """
        # Read from the RSS feed URL
        pod_feed = feedparser.parse(rss_url)
        # Sellect entry/episode and extract info
        self.podcast_title = pod_feed.feed.title
        pod_episode = pod_feed.entries[num]
        self.extractEpisodeInfo(pod_feed, num)
        # Find episode audio URL
        for link in pod_episode.links:
            if link['type'] == 'audio/mpeg':
                episode_url = link.href
        print("RSS URL read. Episode URL: ", episode_url)

        # Setup podcast folder
        podcast_folder = utils.to_filename(self.podcast_title)
        self.podcast_dir = self.content_dir.joinpath(podcast_folder)
        self.podcast_dir.mkdir(exist_ok=True)
        # Setup episode folder
        self.episode_dir = self.podcast_dir.joinpath(f"episode_{self.episode_number}")
        self.episode_dir.mkdir(exist_ok=True)
        # Determin audio, transcript, summary and highlights path
        self.audio_path = self.episode_dir.joinpath("audio.mp3")
        self.transcript_path = self.episode_dir.joinpath("transcript.txt")
        self.summary_path = self.episode_dir.joinpath("summary.txt")
        self.highlights_path = self.episode_dir.joinpath("highlights.txt")

        # Download the podcast audio by parsing the RSS feed.
        print("Downloading the podcast episode ...")
        utils.download_audio(episode_url, self.audio_path)
        print("Podcast episode downloaded.")

    def transcribeAudio(self):
        """
        Loads the audio of the audio at audio_path
        transcribes it and saves it at transcript_path
        """
        # Load transcribing model if it's not already loaded
        if self.trans_model is None:
            raise ImportError("No transcirption model has been loaded.")

        print("Starting podcast transcription ...")
        result = self.trans_model.transcribe(str(self.audio_path))
        print("Transcription completed")

        audio_transcript = result['text']
        utils.save_text(audio_transcript, self.transcript_path)
        print("Transcript saved.")

    def chatComplete(self, prompt, msgs):
        """
        Receives a prompt and the appropriate messages with the roles
        Counts the tokens of the request, and chooses the appropriate model
        Returns the chatComplete from GPT
        """
        enc = tiktoken.encoding_for_model(self.chat_model)
        num_tokens = len(enc.encode(prompt))
        if num_tokens > 4000:
            model_version = "gpt-3.5-turbo-16k"
        else:
            model_version = "gpt-3.5-turbo"

        chatOutput = openai.ChatCompletion.create(model=model_version,
                                                  messages=msgs)
        return chatOutput

    def summarizeTranscript(self):
        """
        Loads the podcast transcript and summarizes it.
        """
        print("Summarizing episode ...")
        # Load podcast transcript
        podcast_transcript = utils.load_text(self.transcript_path)

        instructPrompt = f"""As a podcast enthusiast with limited free time, 
                         I often rely on episode summaries to decide which 
                         podcasts to listen to. I need your assistance in 
                         summarizing an episode from the podcast "{self.podcast_title}" 
                         to help me make an informed choice.

                         I will provide you with the transcript of an episode titled
                          "{self.episode_title}" and a summary provided by the podcasters.
                          Your task is to generate a concise, informative summary of the
                          episode's content based on the transcript. Additionally, please
                          use the summary provided by the podcasters as a reference and
                          consider incorporating relevant information from it into your summary.

                         Transcript: {podcast_transcript}
                         Summary from podcasters: {self.episode_summary}
                         Summary:
                         """

        messages = [{"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": instructPrompt}]

        chatOutput = self.chatComplete(instructPrompt, messages)
        summary = chatOutput.choices[0].message.content
        utils.save_text(summary, self.summary_path)
        print("Summary saved.")

    def getGuestInfo(self, text, info_folder):
        """
        Extracts information about the guest of the podcast.
        Information that we are interested in is his full name,
        the company or the institute he/she is running or is
        part of, and his specialty/job.
        """
        system_role = "You are a helpful assistant that extracts guest information from podcast transcripts."

        msgs = [{"role": "system", "content": system_role},
                {"role": "user", "content": text}]

        function_description = """Get information on the podcast guest using
         their full name and the name of the organization they are part of to
          search for them on Wikipedia or Google
         """

        # Call the OpenAI API to extract guest information
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=msgs,
            functions=[
                {
                    "name": "get_podcast_guest_information",
                    "description": function_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "guest_name": {
                                "type": "string",
                                "description": "The full name of the guest who is speaking in the podcast",
                            },
                            "guest_organization": {
                                "type": "string",
                                "description": """The full name of the organization that
                                                the podcast guest belongs to or runs""",
                            },
                            "guest_title": {
                                "type": "string",
                                "description": """The title, designation or role of the
                                                podcast guest in their organization.""",
                            },
                        },
                        "required": ["guest_name"],
                    },
                }
            ],
            function_call={"name": "get_podcast_guest_information"}
        )

        podcast_guest = ""
        podcast_guest_org = ""
        podcast_guest_title = ""
        response_message = completion["choices"][0]["message"]
        if response_message.get("function_call"):
            function_args = json.loads(response_message["function_call"]["arguments"])
            podcast_guest = function_args.get("guest_name")
            podcast_guest_org = function_args.get("guest_organisation")
            podcast_guest_title = function_args.get("guest_title")

            if podcast_guest_org is None:
                podcast_guest_org = ""
            if podcast_guest_title is None:
                podcast_guest_title = ""

        # Use the guest's info to retrieve more info from wikipedia
        wiki_query = podcast_guest + " " + podcast_guest_org + " " + podcast_guest_title
        if podcast_guest:
            try:
                wiki = wikipedia.page(wiki_query, auto_suggest=True)
                pod_guest_info = wiki.summary
            except:
                print("No wikipedia page found for guest.")
                pod_guest_info = ""

            utils.save_text(pod_guest_info, info_folder + "guest_info.txt")

    def getHighlights(self):
        """
        Extracts the highlights from the podcast's transcript
        """
        print("Getting episode highlights ...")
        podcast_transcript = utils.load_text(self.transcript_path)
        prompt = f"""I'm an avid podcast listener, and I'm always looking for
                 the key takeaways or contentious points discussed in episodes.
                  Additionally, when I listen to self-help podcasts, I like to
                  have a clear list of actionable suggestions and the overarching
                  goal in mind. Can you assist me with this?

                  I'll provide you with the transcript of a podcast episode titled
                  "{self.episode_title}". Your task is to extract the following information:

                  1. **Highlights:** Summarize the key highlights or important insights
                   discussed in the episode.

                  2. **Contentious Points:** Identify and summarize any contentious or debated
                   topics that arise during the episode.

                  3. **Suggestions:** If this is a self-help or advice-based podcast,
                   list the actionable suggestions or advice given by the guest or host,
                    along with the overarching goal they aim to achieve.

                  Transcript: {podcast_transcript}

                  Please organize the information clearly and separate the highlights,
                   contentious points, and self-help suggestions distinctly. This will help
                    me quickly grasp the most important aspects of the episode.
                """

        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]
        chatOutput = self.chatComplete(prompt, msgs)
        highlights = chatOutput.choices[0].message.content
        utils.save_text(highlights, self.highlights_path)
        print("Highlights saved.")

    def makeNewsLetter(self, rss_url):
        """
        Entry-point for news-letter
        """
        self.getPodcast(rss_url)
        self.transcribeAudio()
        self.summarizeTranscript()
        self.getHighlights()


