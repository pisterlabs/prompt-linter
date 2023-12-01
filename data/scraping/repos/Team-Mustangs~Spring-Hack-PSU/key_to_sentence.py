import openai
import time

class ks:
    def __init__(self):
        openai.api_key = "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" 

    # Define a function to ask ChatGPT a question
    def ask_question(self,prompt):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        time.sleep(1) # Pause for a second to avoid hitting the API too quickly
        return response.choices[0].text.strip()

    def ask(self,prompt):
        prompt = "You can't used the work ok or okay in sentences. Genearte simple but factually true sentence from the following keywords: " + prompt 
        answer = self.ask_question(prompt)
        return answer

