import openai
import inspect
from env_vars_parser import get_env_var

class SleepAgent:
    def get_response(self, sleep_data):
        # Create the prompt.
        prompt = inspect.cleandoc(f"""
            Sleep data: {sleep_data}
            """)
        
        system_message = "You are a sleep expert. Can you provide a summary of my past nights' sleep. Then, please analyse the given sleep data to give sleep hygiene suggestions to the patient."
        temperature_value = 0.2
        top_p_value = 1
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        # Set the OpenAI API key.
        openai.api_key = get_env_var("LIFE_UPGRADE_API_KEY")

        # Optional self-consistency approach returns early, selecting the answer choice with the highest frequency
        # return self.__self_consistency_response(messages, answer_choices, temperature_value, top_p_value, 5)

        # Call the OpenAI 3.5 API.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature_value,
            top_p=top_p_value,
        )
        response_text = response.choices[0].message.content

        return response_text