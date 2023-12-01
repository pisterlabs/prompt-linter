from openai import OpenAI
from dotenv import load_dotenv, dotenv_values

load_dotenv()
client = OpenAI(api_key=dotenv_values(".env")["OPENAI_API_KEY"])


def detect_and_fix_bugs(code_base, language):
    # Define an initial message from the user
    messages = [
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ]  # Define messages as a list with a single initial message dictionary

    code_base = code_base
    language = language

    if code_base:
        # Append the user's new message to the messages list
        messages.append({'role': 'user', 'content': f'Please review the following {language} code and identify any bugs or potential issues. If you find any errors, please suggest a fix or improvements to the code: {code_base}'})
        # Create a chat completion using the AI model (assuming 'client' is initialized elsewhere)
        chat_completion = client.chat.completions.create(
            messages=messages,  # Pass the list of messages
            model="gpt-4"  # Use the GPT-4 model for generating a response
        )

        # Retrieve the response content from the chat completion
        # Note: Make sure 'chat_completion' contains the response object with 'choices' available
        reply = chat_completion.choices[0].message.content

        # Add the assistant's response to the messages list
        messages.append({"role": "assistant", "content": reply})
    return reply
