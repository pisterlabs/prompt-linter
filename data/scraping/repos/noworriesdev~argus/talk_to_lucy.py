import os
import openai
import ipdb


def call_lucy_gpt(message, messages):
    print("USER: " + message)
    messages.append({"role": "user", "content": message})

    try:
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::8A0PPeUw",
            messages=messages,
            max_tokens=500,
            temperature=1,
        )
        # ipdb.set_trace()
        messages.append(
            {
                "role": "assistant",
                "content": completion.choices[0]["message"]["content"],
            }
        )
        print("LUCY: " + completion.choices[0]["message"]["content"])
    except Exception as e:
        print(f"Error: {str(e)}")
    return messages


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_KEY")
    messages = []

    def msg(message):
        messages = call_lucy_gpt(message, messages)

    ipdb.set_trace()
    # Example usage:
    # messages = call_lucy_gpt("Hello, Lucy!", messages)
