import base64
import json
from App.Services.OpenAI.OpenAIAssistantManager import OpenAIAssistantManager

class AssistantAPIAdapter:
    def __init__(self):
        self.assistant_manager = OpenAIAssistantManager()

    def generate_open_answer_questions(self, subject, amount_questions):
        assistant_json = self.assistant_manager.load_assistant(
            "Services/OpenAI/Assistants/OpenAnswerQuestionsAssistant.json")
        assistant = self.assistant_manager.create_assistant(assistant_json)

        thread = self.assistant_manager.create_thread()
        messsage = self.assistant_manager.create_message(thread.id, "test")

        run = self.assistant_manager.run_thread(thread.id, assistant.id)
        token = self.encode_token(run.thread_id, run.assistant_id)
        return token

    def generate_multiple_choice_questions(self, subject, amount_questions):
        assistant_json = self.assistant_manager.load_assistant(
            "Services/OpenAI/Assistants/MultipleChoiceQuestionAssistant.json")
        assistant = self.assistant_manager.create_assistant(assistant_json)

        thread = self.assistant_manager.create_thread()
        messsage = self.assistant_manager.create_message(thread.id, "test")

        run = self.assistant_manager.run_thread(thread.id, assistant.id)
        token = self.encode_token(run.thread_id, run.assistant_id)
        return token

    def generate_explanation(self, question, given_answer):
        assistant_json = self.assistant_manager.load_assistant("Services/OpenAI/Assistants/ExplanationAssistant.json")
        assistant = self.assistant_manager.create_assistant(assistant_json)

        thread = self.assistant_manager.create_thread()
        messsage = self.assistant_manager.create_message(thread.id, "test")

        run = self.assistant_manager.run_thread(thread.id, assistant.id)
        token = self.encode_token(run.thread_id, run.assistant_id)
        return token

    def generate_answer(self, question, question_info):
        assistant_json = self.assistant_manager.load_assistant("Services/OpenAI/Assistants/AnswerAssistant.json")
        assistant = self.assistant_manager.create_assistant(assistant_json)

        thread = self.assistant_manager.create_thread()
        messsage = self.assistant_manager.create_message(thread.id, "test")

        run = self.assistant_manager.run_thread(thread.id, assistant.id)
        token = self.encode_token(run.thread_id, run.assistant_id)
        return token

    def retrieve_multiple_choice_questions(self, token):
        thread_id, assistant_id = self.decode_token(token)

        messages = self.assistant_manager.retrieve_messages(thread_id)
        self.assistant_manager.delete_thread(thread_id)
        self.assistant_manager.delete_assistant(assistant_id)

        return messages
    def retrieve_open_answer_questions(self, token):
        thread_id, assistant_id = self.decode_token(token)

        messages = self.assistant_manager.retrieve_messages(thread_id)
        self.assistant_manager.delete_thread(thread_id)
        self.assistant_manager.delete_assistant(assistant_id)

        return messages
    def retrieve_explanation_questions(self, token):
        thread_id, assistant_id = self.decode_token(token)

        messages = self.assistant_manager.retrieve_messages(thread_id)
        self.assistant_manager.delete_thread(thread_id)
        self.assistant_manager.delete_assistant(assistant_id)

        return messages

    def retrieve_answer(self, token):
        thread_id, assistant_id = self.decode_token(token)

        messages = self.assistant_manager.retrieve_messages(thread_id)
        self.assistant_manager.delete_thread(thread_id)
        self.assistant_manager.delete_assistant(assistant_id)

        return messages

    def encode_token(self, thread_id, assistant_id):
        ids_dict = {'thread_id': thread_id, 'assistant_id': assistant_id}
        ids_json = json.dumps(ids_dict)
        ids_bytes = ids_json.encode('utf-8')
        encoded_ids = base64.b64encode(ids_bytes)

        return encoded_ids.decode('utf-8')

    def decode_token(self, token):
        ids_bytes = base64.b64decode(token)
        ids_json = ids_bytes.decode('utf-8')
        ids_dict = json.loads(ids_json)
        return ids_dict['thread_id'], ids_dict['assistant_id']
