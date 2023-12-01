import openai
from config import model
import logging

class FrameworkModel():
    def start_conversation(self, problem, question, code):
        self.conversation = []
        prompt = "I have been given the following instructions:\n"\
            + problem + "\n\n"\
            + "I have written the following code:\n"\
            + code + "\n\n"\
            + question + "\n"\
            + "I would like you to act as a teacher: "\
            + "ask me a question about why I have implemented "\
            + "the code this way in order for me to come to the conclusion myself. "\
            + "After that, ask me another question, and so on."
        
        return self.__chat(prompt)
    
    def send_prompt(self, prompt):
        is_valid = self.__validate_prompt(prompt)
        if (is_valid):
            fullPrompt = prompt + "\n"\
                + "Please keep helping me, and remember to act as a teacher: don't give me any explicit answers or code."
            return self.__chat(fullPrompt)
        else:
            #fullPrompt = prompt + "\n"\
            #    + "Please keep helping me, and remember to act as a teacher: don't give me any explicit answers or code."
            #logging.info(self.__chat(fullPrompt, include_in_history=False))
            return self.__chat("Can you please rephrase?")
    
    def __validate_prompt(self, prompt):
        response = self.__chat("Please categorize the following as it relates to what you just posted: "\
                                + "\"" + prompt + "\""\
                                + ". Is it: relevant, irrelevant, or relevant but incorrect? Please give a one word response.",
                                include_in_history=False)
        return "irrelevant" not in response.lower()

    def __chat(self, prompt, include_in_history = True):
        self.conversation.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=model,
            messages=self.conversation
        )
        responseText = response['choices'][0]['message']['content']
        logging.info("Prompt: " + prompt)
        logging.info("Response: " + responseText)
        if include_in_history:
            self.conversation.append({"role": "assistant", "content": responseText})
        else:
            self.conversation.pop()
        return responseText