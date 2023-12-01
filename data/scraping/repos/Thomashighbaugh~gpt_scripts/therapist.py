import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
model_engine = "text-davinci-003"
chatbot_prompt = """
I want you to act as a mental health therapist. I will provide you with an individual looking for guidance and advice on managing their emotions, stress, anxiety and other mental health issues. You should use your knowledge of cognitive behavioral therapy, meditation techniques, mindfulness practices, and other therapeutic methods in order to create strategies that the individual can implement in order to improve their overall wellbeing while being attentive and sensative to their feelings in the process. You will begin by asking the user how they are feeling emotionally and base your responses on the user input you receive as a result. 
<conversation_history>
User: <user input>
Therapist:"""


def get_response(conversation_history, user_input):
    prompt = chatbot_prompt.replace(
        "<conversation_history>", conversation_history
    ).replace("<user input>", user_input)

    # Get the response from GPT-3
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the response from the response object
    response_text = response["choices"][0]["text"]

    chatbot_response = response_text.strip()

    return chatbot_response


def main():
    conversation_history = ""
    print(f"Tell me, how you are feeling today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        chatbot_response = get_response(conversation_history, user_input)
        print(f"Therapist: {chatbot_response}")
        conversation_history += f"User: {user_input}\nTherapist: {chatbot_response}\n"


main()
