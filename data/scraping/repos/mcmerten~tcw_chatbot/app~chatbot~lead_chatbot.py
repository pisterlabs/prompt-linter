import openai
from app.config import settings
from app.chatbot.prompts import LeadPrompts
from app.core.logger import get_logger

logger = get_logger(__name__)

# Define Chatbot class (Lead Generation Module)
class LeadChatbot:
    def __init__(self, history=None):
        self.chat_history = history or []
        openai.api_key = settings.OPENAI_API_KEY
    # Define ChatCompletion API call
    def get_answer(self, user_message):
        system_prompt = LeadPrompts.system_prompt(self.chat_history)
        try:
            response = openai.ChatCompletion.create(
                # TODO: add config variable for model
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            )
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return

        answer = response['choices'][0]['message']['content']
        if __name__ == "__main__":
            self.chat_history.append((f"User: {user_message}\n", f"Assistant: {answer}\n"))

        # Logic to handle completion / abortion of lead generation process
        if answer == "-1":
            logger.info("Lead generation process aborted.")
            answer = "Was möchten Sie über das TCW wissen?"
            return answer, "Aborted"
        elif answer == "200":
            logger.info("Lead generation process completed successfully.")
            answer = "Vielen Dank für Ihre Informationen. Was möchten Sie über das TCW erfahren?"
            return answer, "Success"
        else:
            return answer, "In Progress"

    def chat(self, query, history=None):
        if history:
            self.chat_history = history
        final_answer, lead_generation_status = self.get_answer(query)
        return final_answer, lead_generation_status

# Define main function to test chatbot
if __name__ == "__main__":
    bot = LeadChatbot()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        response, lead_generation_aborted = bot.chat(user_input)
        print("Bot: ", response)
