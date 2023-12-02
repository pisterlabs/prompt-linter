import os
from typing import List, Dict

import openai

from BookGenerator.PromptGenerator import PromptGenerator


class TextGenerator:
    
    def __init__(self):
        openai.api_key = os.getenv("GPT_API_KEY")

    def getStoriesFromPrompt(self, messages: List[Dict[str, str]], n=1) -> List[Dict[str,str]]:
        """
        Main entry point for TextGenerator, will get a string with a prompt and should return a story that fits the prompt
        :param n:
        :param messages:
        :return:
        """
        print(f"Got request for {n} stories from prompt: {messages}")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # model types: gpt-3.5-turbo, gpt-4-0314, gpt-4, gpt-3.5-turbo-0301
            messages=[{"role": "system", "content": "you are a childerns book writer you write stories up to 70 words"}]
            + messages + [{"role": "system", "content": "Please keep title length under 22 characters long"},
                          {"role": "system", "content": "Please format the title as 'Title:\"<title>\"'"}],
            temperature=1.3,
            max_tokens=1000,
            n=n
        )

        story_lst = []
        for i in range(n):
            story_lst.append(response["choices"][i]["message"]["content"])

        story_list_of_dicts = []
        print("separating stories from titles")
        for story in story_lst:
            splited_lst = story.split('"')
            story_dict = {"title": splited_lst[1], "story": " ".join(splited_lst[2:])[2:]}
            if len(story_dict["title"]) > 50:
                print("Length of title for one of the stories was too long, generating again")
                return self.getStoriesFromPrompt(messages=messages, n=n)
            story_list_of_dicts.append(story_dict)
        return story_list_of_dicts
    


if __name__ == "__main__":
    prompt_gen = PromptGenerator()
    prompt_dict = prompt_gen.getTextPromptFromRequest()
    text_gen = TextGenerator()
    story_output = text_gen.getStoriesFromPrompt(prompt_dict)
    print(story_output)
