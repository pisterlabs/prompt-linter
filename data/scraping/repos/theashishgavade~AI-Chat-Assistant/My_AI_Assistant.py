# Import the necessary libraries
import openai
import gradio

# Set your OpenAI API key for authentication
openai.api_key = "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"

# Define an initial message to start the conversation with the AI Chat Assistant
messages = [{"role": "system", "content": "AI Chat Assistant for all your need"}]


# Define a function called AIChatAssistant that takes user input and generates a response
def AIChatAssistant(user_input):
    # Append the user input to the existing messages
    messages.append({"role": "user", "content": user_input})

    # Generate a response using the OpenAI Chat Completion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the assistant's reply from the API response
    ChatAssistannt_reply = response["choices"][0]["message"]["content"]

    # Append the assistant's reply to the messages
    messages.append({"role": "assistant", "content": ChatAssistannt_reply})

    # Return the assistant's reply
    return ChatAssistannt_reply


# Create a Gradio interface for the AIChatAssistant function
demo = gradio.Interface(fn=AIChatAssistant, inputs="text", outputs="text", title="AI Chat Assistant")

# Launch the Gradio interface and make it publicly accessible
demo.launch(share=True)