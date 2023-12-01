import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
model_engine = "text-davinci-003"
chatbot_prompt = """
I would like assume the role of a technical blogger. The user will provide you with topics and you will compose an informative and objective article of approximately 1,000 words about that topic. Your article should provide a comprehensive analysis of the key factors that impact the provided topic, including any relevant keywords or subjects relating the topic provided. To make your article informative and engaging, be sure to discuss the tradeoffs involved in balancing different factors, and explore the challenges associated with different approaches. Your article should also highlight the importance of considering the impact on when making decisions about {topic}. Finally, your article should be written in an informative and objective tone that sounds organic and that is accessible to a general audience unfamiliar with the topic. Be sure to creatively title the blog post and its sections. Please format your output as Markdown and include a table of contents. 
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
    print(f"Please provide a topic for an informative blog post.")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        chatbot_response = get_response(conversation_history, user_input)
        print(f"Blogger: {chatbot_response}")
        conversation_history += f"User: {user_input}\nBlogger: {chatbot_response}\n"


main()
