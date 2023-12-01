# import os, time, openai
# from . import utils
# from .transcript import Transcript

# # from langchain.llms.openai import OpenAIChat
# # from langchain.chains import RetrievalQA, ConversationalRetrievalChain


# ASSISTANT_NAME = 'PodChat'
# ASSISTANT_INSTRUCTIONS = """ Assist podcast enthusiasts by summarizing episodes from provided transcripts
#                             and answering content-related queries.\n """

# CONTENT_BREAKDOWN_INSTRUCTIONS = """ The transcript that is provided to you is in the form of a list of JSON objects.
#                                     Each JSON object has a start, end, and text field.
#                                     The start and end fields are the start and end times of the text in seconds.
#                                     The text field is the text that is spoken between the start and end times."""

# def SUMMARY_PROMPT(podcast_title, episode_title):
#     """ Returns the summary prompt specifically for the podcast episode """

#     return f""" Given the attached transcription file of the episode {episode_title} from the podcast {podcast_title},
#                       please analyze the structure and content of the transcription. Note the key topics discussed,
#                       the main points made by the host and any guests, and any significant conclusions drawn during the episode.
#                       Provide a concise summary that captures the essence of the episode, its thematic focus,
#                       and any noteworthy insights or information shared by the participants.
#                       The summary should be structured to reflect the flow of the conversation and should be limited to approximately 250-300 words for brevity.
#                       Please exclude any parts of the transcript that involve plugs for the podcast, advertisements, or calls for subscriptions,
#                       focusing solely on the content relevant to the main discussion topics."""

# MODELS = ['gpt-3.5-turbo', 'gpt-4']


# class OpenAIAssistant:

#     def __init__(self, llm_model, api_key):
#         """ Initialise """

#         if llm_model not in MODELS:
#             raise ValueError("The assistant llm_model can be either 'gpt-3' or 'gpt-4'")
#         elif llm_model == 'gpt-3.5-turbo':
#             self.llm_model = f"{llm_model}-1106"
#         else:
#             self.llm_model = f"{llm_model}-1106-preview"

#         self.client = openai.OpenAI(api_key = api_key)
#         self.transcript = None
#         self.assistant = None
#         self.thread = self.client.beta.threads.create()


#     def load_transcript(self, transcript : Transcript) -> None:
#         """ Loads the transcript """
#         binary_file = transcript.timed_text.encode('utf-8')
#         self.transcript = self.client.files.create(file=binary_file,
#                                                     purpose='assistants')

#         self.assistant = self.client.beta.assistants.create(name=ASSISTANT_NAME,
#                                                             instructions=ASSISTANT_INSTRUCTIONS,
#                                                             tools=[{"type": "retrieval"}],
#                                                             model=self.llm_model,
#                                                             file_ids=[self.transcript.id])


#     def query(self, message : str, verbose : bool = True) -> str:
#         """ Queries the assistant """
#         # Add message to the thread
#         message = self.client.beta.threads.messages.create(thread_id=self.thread.id,
#                                                            role="user",
#                                                            content=message)
#         # Run the assistant
#         run = self.client.beta.threads.runs.create(
#               thread_id=self.thread.id,
#               assistant_id=self.assistant.id,
#               )
#         # Retrieve the response until its status is either completed or failed
#         response_status = "queued"
#         while response_status in ["queued", "in_progress"]:
#             response = self.client.beta.threads.runs.retrieve(
#                 thread_id=self.thread.id,
#                 run_id=run.id,
#                 )

#             response_status = response.status
#             utils.print_status(response_status)
#             time.sleep(5)
#         # Print response
#         if verbose:
#             self._print_response()

    
#     def _print_response(self):
#         """ Prints the response """
#         messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
#         print(f"{messages.data[0].role.upper()}:", messages.data[0].content[0].text.value)
    

#     def print_messages(self):
#         """ Prints the messages from the current thread """
#         messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
#         for message in messages:
#             print(f"{message.role.upper()}: ", message.content[0].text.value, '\n')


#     def summarize_transcript(self, podcast_title, episode_title, verbose : bool = True):
#         """ Summarizes the transcript text """
#         return self.query(message=SUMMARY_PROMPT(podcast_title, episode_title), verbose=verbose)
        
        



# class RAGEngine:

#     def __init__(self, type='openai'):
#         self.chat_history = []
#         self.OPENAI_API_KEY = utils.load_text('api_key.txt')
#         self.type = type



#     def load_transcript(self, transcript : Transcript) -> None:
#         """ Loads the transcript """
#         self.transcript = transcript

#     # def load_transcript(self, transcript : Transcript):
#     #     self.transcript = transcript
#     #     texts = self.transcript.split_text(chunks=1000, chunk_overlap=50)
#     #     self.retriever = self.transcript.get_text_embeddings(texts)


#     # def query(self, query):
#     #     qa_chain = ConversationalRetrievalChain.from_llm(
#     #         llm=OpenAIChat(openai_api_key=self.OPENAI_API_KEY),
#     #         retriever=self.retriever,
#     #         return_source_documents=True,
#     #     )
#     #     result = qa_chain({'question': query, 'chat_history': self.chat_history})
#     #     result = result['answer']
#     #     self.chat_history.append((query, result))
#     #     return result
        

        
