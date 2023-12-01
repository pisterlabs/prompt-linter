import os
import logging
import time # for typing simulation
import threading # for typing simulation
from dotenv import load_dotenv # .env file
from database_manager import DatabaseManager

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from langchain.text_splitter import CharacterTextSplitter # split text into chunks
from langchain.embeddings import OpenAIEmbeddings # vector store
from langchain.vectorstores import FAISS # vector store database
from langchain.chat_models import ChatOpenAI # conversational memory
from langchain.memory import ConversationBufferMemory # conversational memory
from langchain.chains import ConversationalRetrievalChain # conversational memory

from metaphor_python import Metaphor # evolve into agent

class SlackBot:

    def start(self): # Start the Slack bot
        handler = SocketModeHandler(self.app, self.SOCKET_TOKEN)

        # Fetch the list of channels
        channels = self.app.client.conversations_list(limit=1000)  # Adjust the limit as needed
        general_channel = next((channel['id'] for channel in channels['channels'] if channel['name'] == 'general'), None)

        # Welcome Message in the last active channel or the general channel
        last_channel = self.db_manager.get_last_active_channel()
        channel_to_send = last_channel if last_channel else general_channel
        self.app.client.chat_postMessage(channel=channel_to_send, text="Your personal training session has started! 💪🏻🔥")

        handler.start() # initiates bot, allowing listening

    def __init__(self):
        load_dotenv() # Load environment variables
        self.SLACK_BOT_USER_ID = os.environ.get("SLACK_BOT_USER_ID")
        self.SOCKET_TOKEN = os.environ.get("SOCKET_TOKEN")
        self.SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

        # Initialize the Slack/DB/Metaphor
        self.app = App(token=self.SLACK_BOT_TOKEN)
        self.db_manager = DatabaseManager()
        METAPHOR_API_KEY = os.environ.get("METAPHOR_API_KEY")
        self.metaphor = Metaphor(METAPHOR_API_KEY)

        # Setup event handler
        @self.app.event("app_mention")
        def handle_mentions(body, say):
            user_id = body["event"]["user"]  # The user who triggered the event
            channel_id = body["event"]["channel"]  # The channel where the event was triggered
            self.db_manager.update_last_active_channel(channel_id)  # Update the last active channel in the database

            # Post the initial "Working on it" message
            initial_message = self.app.client.chat_postMessage(channel=channel_id, text="😁 Working on it")
            timestamp = initial_message['ts']  # Store the timestamp of the message

            # This function simulates the "typing" effect by updating the message
            def typing_simulation(channel_id, timestamp, stop_event):
                dot_count = 0
                while not stop_event.is_set():  # Loop until the event is set
                    time.sleep(0.1)
                    dot_count += 1
                    self.app.client.chat_update(channel=channel_id, ts=timestamp, text=f"😁 Working on it{'.' * dot_count}")
                    if dot_count == 3:
                        dot_count = -1

            stop_typing_event = threading.Event()  # Event to signal the typing simulation to stop
            typing_thread = threading.Thread(target=typing_simulation, args=(channel_id, timestamp, stop_typing_event))
            typing_thread.start()  # Start the typing simulation

            # Add the previous conversation history to the memory
            text = body["event"]["text"]
            mention = f"<@{self.SLACK_BOT_USER_ID}>"
            text = text.replace(mention, "").strip()

            # Add the instruction to the beginning of the text
            instruction = "You are a bodybuilding coach named Muscle Mentor. Talk in a positive, cheerful tone. Always give accurante answers. Never give false information. "
            preprompted_text = instruction + "\n" + text

            # Check for video requests
            video_response = self.get_response(preprompted_text)
            if video_response:
                response_text = video_response
                response = {'answer': video_response}  # Set the 'answer' key to the video_response
            else:
                # Process the user's question and generate a response using the conversation chain
                response = self.conversation_chain({'question': preprompted_text})
                response_text = response.get('answer', '')  # Use .get() to avoid KeyError

            stop_typing_event.set()  # Signal the typing simulation to stop
            typing_thread.join()  # Wait for the typing simulation to finish

            # Replace the "Working on it..." message with the actual response
            self.app.client.chat_update(channel=channel_id, ts=timestamp, text=response_text)

            # Fetch the previous conversation history from the database
            previous_history = self.db_manager.get_conversation_history(user_id) or ""

            # Update the conversation history in the database
            updated_history = previous_history + "\n" + text + "\n" + response['answer']
            self.db_manager.update_conversation_history(user_id, updated_history)

        # Setup the bot
        self.setup_bot()

    def setup_bot(self):
        documents = self.load_and_transform_data()
        document_chunks = self.split_documents_into_chunks(documents)
        vectorstore = self.get_vectorstore(document_chunks)
        self.conversation_chain = self.get_conversation_chain(vectorstore)

    def load_and_transform_data(self):
        """Loads the text files from the directory and returns them as a list of dictionaries."""
        directory_path = os.getenv("CUSTOM_KNOWLEDGE_DIRECTORY")
        all_documents = []

        for text_file in os.listdir(directory_path):
            if text_file.endswith('.txt'):
                full_file_path = os.path.join(directory_path, text_file)
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_documents.append({'title': text_file, 'content': content})

        return all_documents

    def split_documents_into_chunks(self, documents):
        """Splits the documents into chunks and returns them as a list of dictionaries."""
        text_splitter = CharacterTextSplitter(separator="", chunk_size=1000, chunk_overlap=200, length_function=len)
        all_chunks = []

        for doc in documents:
            chunks = text_splitter.split_text(doc['content'])
            all_chunks.extend([{'title': doc['title'], 'content': chunk} for chunk in chunks])

        return all_chunks

    def get_vectorstore(self, document_chunks):
        """Creates and returns a vector store from the provided document chunks."""
        embeddings = OpenAIEmbeddings()
        texts = [doc['content'] for doc in document_chunks]
        vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)

        return vectorstore

    def get_conversation_chain(self, vectorstore):
        """Initializes and returns a conversation chain."""\
        """Takes the history, and returns the next element in the chain."""
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory)

        return conversation_chain
    
    def get_response(self, text):
        # Check if the user is asking for a video
        if any(keyword in text.lower() for keyword in ["video", "youtube", "link"]):
            # Transform the user's question into a search-friendly query
            search_query = "Find youtube links to " + text
            try:
                response = self.metaphor.search(
                    search_query,
                    num_results=1,
                    include_domains=["https://www.youtube.com"],
                    use_autoprompt=True,
                )
                # Extract the video URL and return it
                video_url = response.results[0].url
                return f"Here's a video I found: {video_url}"

            except Exception as e:
                logging.error(f"Error searching for video: {e}")
                return "Sorry, I couldn't find a video for that."

        # If not asking for a video, proceed with regular processing
        return None