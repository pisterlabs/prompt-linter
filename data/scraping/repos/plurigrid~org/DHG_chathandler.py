import openai
import json

from DHG_llm import ConversationDHGConstructor

class GPTChat:
    def __init__(self, dhg_name):
        self.dhg_name = dhg_name
        self.dhg = ConversationDHGConstructor(self.dhg_name)
    
    def chat_with_gpt(self):
        while True:
            user_message = input("")
            
            if user_message.lower() == 'exit':
                print("Exiting conversation...")
                break

            self.dhg.add_message("User", user_message)

            gpt_response = self.get_gpt_response(user_message)
            
            self.dhg.add_message("GPT", gpt_response)
            print(f"GPT: {gpt_response}")
        
        self.dhg.save_to_file()

    def get_gpt_response(self, prompt):
        openai.api_key = 'your-api-key'

        try:
            response = openai.Completion.create(
              engine = "text-davinci-003",
              prompt = prompt,
              max_tokens = 150
            )
            
            return response.choices[0].text.strip()
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return "I'm sorry, but I couldn't generate a response."

if __name__ == "__main__":
    chat = GPTChat("conversation")
    chat.chat_with_gpt()
