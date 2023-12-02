import os
from lib.ui_messages import UIMessages as UI
from lib.messenger import Messenger as msg
from lib.dynamodb import DynamoDB
from lib.chat_model import ChatModel
from lib.tutor_chain import TutorChain
from lib.tutor_prompt_builder import TutorPromptBuilder
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory import ConversationTokenBufferMemory

def save_conversation(db, message, response):
    db.add_user_message(message)
    db.add_ai_message(response)


# Create a Menu class that will be used to create a menu for the bot
class Menu:
    def __init__(self):
        self.config = {
            "TELEGRAM_TOKEN": os.environ['TELEGRAM_TOKEN'],
            "OPENAI_API_KEY": os.environ['OPENAI_API_KEY'],
            "USERS_TABLENAME": os.environ['USERS_TABLENAME'],
            "CHAT_MESSAGES_TABLENAME": os.environ['CHAT_MESSAGES_TABLENAME']
        }

        self.msg = msg(self.config)
        self.users_table = DynamoDB(self.config.get('USERS_TABLENAME'))

    def start(self, chat_info):
        """Add user to DynamoDB table and send welcome message"""

        chat_id = chat_info['id']
        self.users_table.new_user(chat_info)
        message = UI.welcome_message(chat_info.get('first_name', "student"))
        self.msg.send_to_telegram(message, chat_id)

    def help(self, chat_id):
        """Send help message"""

        message= UI.help_message
        self.msg.send_to_telegram(message, chat_id)

    def about(self, chat_id):
        """Send about message"""

        message = UI.about_message
        self.msg.send_to_telegram(message, chat_id)

    def subject(self, chat_id, message):
        """Change subject or assign a new subject to user"""

        if message == "/subject":
            message = UI.no_subject_specified_message

            self.msg.send_to_telegram(message, chat_id)
            return

        subject = ' '.join(message.split(' ')[1:])

        if subject == "":
            message = UI.no_subject_message
            self.msg.send_to_telegram(message, chat_id)
            return
        
        message = UI.subject_message(subject)
        self.users_table.update_subject(chat_id, subject)
        self.msg.send_to_telegram(message, chat_id)

    def evaluate():
        """Evaluate user about a subject. Not Implemented yet."""
        pass

    def interaction(self, chat_id, message):
        """Start an interaction with the bot. If user has a subject, bot will respond to user's message."""

        user = self.users_table.get_user(chat_id)

        if 'subject' in user:

            session_id = str(chat_id) + user['subject']
            history = DynamoDBChatMessageHistory(table_name=self.config.get('CHAT_MESSAGES_TABLENAME'),
                                                 session_id=session_id)

            template = TutorPromptBuilder.template(user)
            chat_prompt = TutorPromptBuilder.build(template)

            
            chat = ChatModel(openai_api_key=self.config['OPENAI_API_KEY'],
                             temperature=1.2,
                             model="gpt-3.5-turbo-0613")
            
            memory = ConversationTokenBufferMemory(memory_key="chat_history",
                                              chat_memory=history,
                                              llm=chat,
                                              return_messages=True,
                                              max_token_limit=1000)
            
            langchain = TutorChain(llm=chat,
                                   prompt=chat_prompt,
                                   memory=memory)
            response = langchain.run(message)

            self.msg.send_to_telegram(response, chat_id)

            save_conversation(history, message, response)
            
        elif user == "No user found.":
            message = UI.no_user_found_message
            self.msg.send_to_telegram(message, chat_id)
            return
        else:
            message = UI.select_subject_message
            self.msg.send_to_telegram(message, chat_id) 
        return

    def no_text(self, chat_id):
        """Respond to non-text messages"""

        message= UI.no_text_message
        self.msg.send_to_telegram(message, chat_id)
