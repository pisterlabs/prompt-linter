import logging
import json
import os
import time
import datetime
import openai
from dotenv import load_dotenv
from Prompt import julie_description
from .models import Chat, CustomUser
from .brain import LongTermMemory

load_dotenv('keys.env')
client = openai.OpenAI()

logging.basicConfig(filename='juliebot.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class Juliebot:
    """
A class representing a chatbot assistant named Juliebot.

Attributes:
    long_term_memory (LongTermMemory):
    The long-term memory object used by Juliebot.

Methods:
    create_assistant(user_name):
    Creates a new assistant for the specified user.
    create_thread(): Creates a new thread using the client's beta API.
    manage_chat_session(user_name, new_message, role):
    Manages the chat session for a user.
    send_message(task, user_name):
    Sends a message from the user to the chatbot.
    run_assistant(user_name): Runs the assistant for the given user.
    process_run_output(user_name, chat_session):
    Process the run output for a chat session.
    display_sessions(file_path='chat_sessions.json'):
    Display the available sessions from the specified file.
    get_session_data(session_number, file_path='chat_sessions.json'):
    Retrieves session data for a given session number from a JSON file.
    collect_message_history(user_name):
    Collects the message history for a given user and saves it to a text file.
"""
    def __init__(self, long_term_memory: LongTermMemory):
        self.long_term_memory = long_term_memory

    def create_assistant(self, user_name):
        """
        Creates a new assistant for the specified user.

        Args:
            user_name (str): The name of the user.

        Returns:
            str: The ID of the newly created assistant.
        """
        # Format the description with the username
        formatted_description = julie_description.format_description(user_name)

        # Convert the formatted description to a string
        instructions = formatted_description.string_maker()

        assistant = client.beta.assistants.create(
            name=user_name,
            instructions=instructions,
            model="gpt-4-1106-preview",
        )
        logging.info("New assistant created for user %s", user_name)
        return assistant.id

    def create_thread(self):
        """
        Creates a new thread using the client's beta API.

        Returns:
            str: The ID of the newly created thread.
        """
        thread = client.beta.threads.create()
        logging.info("New thread created")
        return thread.id

    def manage_chat_session(self, user_name, new_message, role):
        """
        Manages the chat session for a user.

        Args:
            user_name (str): The username of the user.
            new_message (str): The new message to be added to the chat history.
            role (str): The role of the user in the chat session.

        Returns:
            Chat: The updated chat session object.

        Raises:
            ValueError: If the user is not found.
        """
        user = CustomUser.objects.filter(username=user_name).first()
        if not user:
            logging.error("No user found for username %s", user_name)
            raise ValueError("User not found")

        chat_session, created = Chat.objects.get_or_create(user=user)
        if created or not chat_session.assistant_id:
            chat_session.assistant_id = self.create_assistant(user_name)
        if created or not chat_session.thread_id:
            chat_session.thread_id = self.create_thread()

        chat_history = chat_session.messages
        timestamp = datetime.datetime.now().isoformat()
        chat_history.append({"role": role, "message": new_message,
                             "timestamp": timestamp})
        chat_session.messages = chat_history
        chat_session.save()
        return chat_session

    def send_message(self, task, user_name):
        """
        Sends a message from the user to the chatbot.

        Args:
            task (str): The message content.
            user_name (str): The name of the user.

        Raises:
            Exception: If there is an error while sending the message.

        Returns:
            None
        """
        try:
            chat_session = self.manage_chat_session(user_name, task, "user")
            self.long_term_memory.update_conversation_history(
                user_name, "user", task)
            client.beta.threads.messages.create(
                thread_id=chat_session.thread_id,
                role="user",
                content=task,
            )
            logging.info("Message sent for user %s: %s", user_name, task)
        except Exception as e:
            logging.error("Error in send_message for user %s: %s", user_name,
                          e)
            raise e

    def run_assistant(self, user_name):
        """
        Runs the assistant for the given user.

        Args:
            user_name (str): The username of the user.

        Returns:
            str: The response generated by the assistant,
            or a message indicating no response was generated.

        Raises:
            Exception: If there is an error while running the assistant.
        """
        try:
            chat_session = Chat.objects.get(user__username=user_name)
            run = client.beta.threads.runs.create(
                thread_id=chat_session.thread_id,
                assistant_id=chat_session.assistant_id
            )
            while run.status in ["in_progress", "queued"]:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=chat_session.thread_id,
                    run_id=run.id
                )
            if run.status == "completed":
                return self.process_run_output(user_name, chat_session)
            else:
                logging.info("""No response generated by the assistant
                             for user %s""", user_name)
                return "No response generated by the assistant."
        except Exception as e:
            logging.error("""Error in run_assistant for user %s: %s""",
                          user_name, e)
            raise e

    def process_run_output(self, user_name, chat_session):
        """
        Process the run output for a chat session.

        Args:
            user_name (str): The name of the user.
            chat_session (ChatSession): The chat session object.

        Returns:
            str: The assistant's response retrieved from the run output.

        Raises:
            Exception: If there is an error processing the run output.
        """
        try:
            messages = client.beta.threads.messages.list(
                thread_id=chat_session.thread_id)
            sorted_messages = sorted(messages.data, key=lambda x:
                                     x.created_at, reverse=True)
            for message in sorted_messages:
                if message.role == "assistant":
                    text_content = message.content[0].text.value
                    self.manage_chat_session(user_name, text_content,
                                             """assistant""")
                    self.long_term_memory.update_conversation_history(
                        user_name, "assistant", text_content)
                    logging.info("Assistant response retrieved: %s",
                                 text_content)
                    return text_content
        except Exception as e:
            logging.error("Error processing run output for user %s: %s",
                          user_name, e)
            raise e

    def display_sessions(self, file_path='chat_sessions.json'):
        """
        Display the available sessions from the specified file.

        Args:
            file_path (str): The path to the sessions file. Default is
            'chat_sessions.json'.

        Returns:
            None
        """
        if not os.path.exists(file_path):
            print("No sessions available.")
            logging.info("No sessions file found for display_sessions")
            return
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print("Available Sessions:")
        for number, session in data["sessions"].items():
            print(f"Session {number}: {session['User Name']}")
        logging.info("Sessions displayed")

    def get_session_data(self, session_number, file_path='chat_sessions.json'):
        """
        Retrieves session data for a given session number from a JSON file.

        Args:
            session_number (int): The session number to retrieve data for.
            file_path (str, optional): The file path of the JSON file.
            Defaults to 'chat_sessions.json'.

        Returns:
            str or None: The user name associated with the session if found,
            None otherwise.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        session = data["sessions"].get(session_number)
        if session:
            self.assistant_id = session["Assistant ID"]
            self.thread_id = session["Thread ID"]
            logging.info("Session data retrieved for session %s",
                         session_number)
            return session["User Name"]
        else:
            print("Session not found.")
            logging.warning("Session %s not found in get_session_data",
                            session_number)
            return None

    def collect_message_history(self, user_name):
        """
        Collects the message history for a given user and
        saves it to a text file.

        Args:
            user_name (str): The name of the user.

        Returns:
            str: A message indicating the file path where
            the messages are saved.
        """
        messages = client.beta.threads.messages.list(thread_id=self.thread_id)
        message_dict = json.loads(messages.model_dump_json())
        with open(f'{user_name}_message_log.txt', 'w',
                  encoding='utf-8') as message_log:
            for message in reversed(message_dict['data']):
                text_value = message['content']
                prefix = "You: " if message['role'] == 'user' else f"""
                {user_name}:"""
                message_log.write(f"{prefix} {text_value}\n")
        logging.info("Message history collected for user %s", user_name)
        return f"Messages saved to {user_name}_message_log.txt"
