import openai
class think:
    def __init__(self):
        pass
    def run(self, string):
        print("Thinking:")
        result = self.chat_with_gpt3(string)
        return result
    

    
    @staticmethod
    def chat_with_gpt3(prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "I am an artifical cognitive entity. I need to think about something. Only reply with a thought in first person."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9
        )
        return response['choices'][0]['message']['content']
