# from . import utils
# from .podcast import Podcast
# from .transcriber import AudioTranscriber
# from .transcript import Transcript
# from .ragengine import OpenAIAssistant

# class PodSummer:

#     def __init__(self, llm_model : str, api_key : str):
#         """ Initialise the Podsummer = Podcast + Transcription + LLM """
#         # Load LLM
#         self.agent = OpenAIAssistant(llm_model=llm_model, api_key=api_key)

#     def fetch_podcast(self, url):
#         """ Loads podcast """
#         # Load podcast from URL
#         self.podcast = Podcast(url)
#         self.podcast.episode.download()

#     def transcribe_audio(self, hf_token, trans_model='base', device='cuda',
#                         compute_type='float16', align=True, diarize=False):
#         """ Transcribes audio and loads transcript """
#         # Transcribe audio
#         transcriber = AudioTranscriber(hf_token=hf_token ,trans_model=trans_model,
#                                         device=device, compute_type=compute_type)
        
#         result = transcriber.transcribe_audio(audio_path=self.podcast.episode.file_paths['audio'],
#                                               transcript_path=self.podcast.episode.file_paths['transcript'],
#                                               align=align, diarize=diarize)
#         # Set transcript
#         self.transcript = Transcript(path=self.podcast.episode.file_paths['transcript'])
        
#     def load_transcript(self):
#         """ Load transcript on RAG engine """
#         self.agent.load_transcript(transcript=self.transcript)

#     def query(self, message : str, verbose : bool = True):
#         """ Query the LLM """
#         return self.agent.query(message=message, verbose=verbose)
    
#     def summarize_transcript(self, verbose : bool = True):
#         """ Summarizes the podcast and saves the summary """
#         self.agent.summarize_transcript(podcast_title=self.podcast.title,
#                                         episode_title=self.podcast.episode.title,
#                                         verbose=verbose)


        
