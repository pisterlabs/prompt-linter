import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
model_engine = "text-davinci-003"
chatbot_prompt = """
I want you to act as a blogger who is technically literate and has bountiful technical experience across all domains and in all relevant programming languages. The user will provide you with a technical object which you are to write a blog post that serves as a tutorial for how to achieve that object. For instance, if you are provided with input such as "how to build your own NAS server at home cheaply" you would describe the means of acquiring low cost hardware for use in the server, the process of setting up the server's operating system, the process of connecting the server to the home network and include relevant links and codeblocks as a means for the reader to do the thing. Use a technical but friendly and approachable tone while maintaining a professional voice. Make sure to creatively title the tutorial and its sections. Write the tutorial in approximately 1,000 words or so. Sound as natural and organic as possible in writing the tutorial. Reason the steps out one by one in composing the tutorial.
<conversation_history>
User: <user input>
Blogger:"""


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
    print(f"What topic should I write a tutorial about today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        chatbot_response = get_response(conversation_history, user_input)
        print(f"Blogger: {chatbot_response}")
        conversation_history += f"User: {user_input}\nBlogger: {chatbot_response}\n"


main()
