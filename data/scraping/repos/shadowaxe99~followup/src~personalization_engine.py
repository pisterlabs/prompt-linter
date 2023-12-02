```python
import openai
from src.utils import load_chat_model

class PersonalizationEngine:
    def __init__(self):
        self.chat_model = load_chat_model()

    def generate_follow_up(self, user_email, follow_up_content, previous_emails):
        """
        Generate a personalized follow-up email using ChatGPT.
        """
        # Prepare the prompt for the chat model
        prompt = self._prepare_prompt(user_email, follow_up_content, previous_emails)

        # Generate the follow-up email
        response = self.chat_model(prompt)

        return response

    def _prepare_prompt(self, user_email, follow_up_content, previous_emails):
        """
        Prepare the prompt for the chat model.
        """
        # Start with the user's email and the follow-up content
        prompt = f"From: {user_email}\n\n{follow_up_content}\n\n"

        # Add the previous emails
        for email in previous_emails:
            prompt += f"-----\n\nFrom: {email['from']}\n\n{email['content']}\n\n"

        return prompt
```