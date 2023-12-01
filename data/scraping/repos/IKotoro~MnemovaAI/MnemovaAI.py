import openai
import os

# Replace 'your_api_key' with your actual API key
API_KEY = 'your_api_key'

# Function to read the prompt from a text file
def read_prompt(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        return None

def send_message(messages, api_key):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        api_key=api_key
    )
    return response.choices[0].message.content

def save_conversation(conversation, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for message in conversation:
            file.write(f"{message['role']}: {message['content']}\n")

def load_conversation(file_name):
    conversation = []
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split(": ", 1)
            if len(parts) == 2:
                role, content = parts
                conversation.append({"role": role, "content": content})
    return conversation

def clear_conversation(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)

def main():
    conversation_file = "conversation.txt"
    prompt_file = "prompt.txt"

    # Read the prompt from the prompt.txt file
    prompt = read_prompt(prompt_file)

    if not prompt:
        print("Please create a prompt.txt file with the desired prompt.")
        return

    if os.path.exists(conversation_file):
        conversation = load_conversation(conversation_file)
    else:
        conversation = [{"role": "system", "content": prompt}]

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            break  # Exit the loop

        if user_input.lower() == 'deconv':
            clear_conversation(conversation_file)
            print("Conversation cleared.")
            continue

        conversation.append({"role": "user", "content": user_input})

        # Send the conversation to the ChatGPT model
        assistant_reply = send_message(conversation, API_KEY)

        print("AI: " + assistant_reply)

        # Append the assistant's reply to the conversation
        conversation.append({"role": "assistant", "content": assistant_reply})

        # Save the conversation to the file
        save_conversation(conversation, conversation_file)

if __name__ == "__main__":
    main()

