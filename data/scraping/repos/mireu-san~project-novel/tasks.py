# celeryapp/tasks.py
from celery import shared_task
import openai


@shared_task
def process_openai_request(user_message):
    predefined_prompt = {
        "role": "system",
        "content": "You are an anime expert. Your role is to listen to a user input and based on his/her expression, suggest any anime, light novel, visual novel or manga for this user.",
    }
    user_input = {"role": "user", "content": user_message}
    messages = [predefined_prompt, user_input]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        response_text = response["choices"][0]["message"]["content"]
        return response_text
    except openai.error.OpenAIError as e:
        error_message = f"OpenAI error: {str(e)}, error details: {e.error}"
        return error_message  # Handle error accordingly
    except Exception as e:
        error_message = f"Error processing OpenAI request: {str(e)}"
        return error_message  # Handle error accordingly
