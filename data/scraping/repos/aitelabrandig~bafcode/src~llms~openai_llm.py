import openai
from core import BafLog
from config import Config

class OpenAILLM:
    def __init__(self):
        self.logger = BafLog

        # Initialize OpenAI API key (should be kept secret and ideally loaded from a secure environment)
        openai.api_key = Config.OPENAI_API_KEY

    def process(self,message,prompt):
    
        if not prompt:
            self.logger.error("No prompt provided for OpenAI LLM.")
            raise ValueError("A prompt is required for processing.")

        try:
            # Use OpenAI's Completion API to get the model's response
            response = openai.ChatCompletion.create(
                model=Config.OPENAI_ENGINE,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message}
                ]
                )
            print(response.choices[0].message.content)
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error processing with OpenAI LLM: {str(e)}")
            return {
                'message': "Error processing with OpenAI LLM.",
                'status': "error"
            }
