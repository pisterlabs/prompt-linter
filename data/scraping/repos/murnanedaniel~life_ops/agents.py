import openai

class PromptManager:
    def __init__(self, prompt_dir='prompts'):
        self.prompt_dir = prompt_dir

    def load_prompt(self, prompt_name):
        with open(f"{self.prompt_dir}/{prompt_name}.txt", 'r') as f:
            return f.read()
import openai


class ChatManager:
    def __init__(self, api_key, prompt_manager):
        self.api_key = api_key
        self.prompt_manager = prompt_manager

    def chat(self, system_prompt, user_prompt, model="gpt-4"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message['content']

    def review_request(self, content, life_document):
        system_prompt = self.prompt_manager.load_prompt('life_coach_prompt').format(life_document=life_document)
        user_prompt = self.prompt_manager.load_prompt('review_user_prompt').format(content=content)
        response = self.chat(system_prompt, user_prompt)
        # Parsing logic here
        return review, approved

    def ai_merge(self, merge_request, current_content):
        system_prompt = self.prompt_manager.load_prompt('merge_manager_prompt')
        user_prompt = self.prompt_manager.load_prompt('merge_user_prompt').format(current_content=current_content, merge_request=merge_request)
        merged = self.chat(system_prompt, user_prompt)
        return merged.content
