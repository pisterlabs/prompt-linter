import openai

# Set up OpenAI API credentials
openai.api_key = "YOUR_API_KEY"

# Define the conversation history
conversation = [
    {"role": "system", "content": "You are chatting with Andrew Tate."},
    {"role": "user", "content": "Hello, Andrew!"},
]

# Main loop for user interaction
while True:
    # Prompt user for input
    user_input = input("You: ")

    # Add user message to the conversation history
    conversation.append({"role": "user", "content": user_input})

    # Generate response from the chatbot
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=conversation,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )

    # Extract the chatbot's reply
    chatbot_reply = response.choices[0].text.strip()

    # Add chatbot's reply to the conversation history
    conversation.append({"role": "assistant", "content": chatbot_reply})

    # Display chatbot's response
    print("Andrew Tate: " + chatbot_reply)
