import openai
from openai import Completion
import tkinter as tk

# Set up the OpenAI API client
openai.api_key = "YOUR API KEY"

# Create the main window
root = tk.Tk()
root.title("ChatGPT App")
root.configure(bg='light blue')

# Create a text entry field and a send button
entry = tk.Entry(root, font=('Arial', 14), bg='white', fg='black')

# Create a text widget to display the conversation
conversation = tk.Text(root, font=('Arial', 14), bg='white', fg='black')

# Create a scrollbar for the conversation widget
scrollbar = tk.Scrollbar(root, orient='vertical', command=conversation.yview)
conversation['yscrollcommand'] = scrollbar.set

# Create a function to clear the conversation widget
def clear_conversation():
    conversation.delete('1.0', 'end')

# Create a function to send a message to ChatGPT and display the response
def send_message():
    # Get the message from the entry field
    message = entry.get()

    # Clear the entry field
    entry.delete(0, "end")

    # Use the OpenAI API to get a response from ChatGPT
    response = openai.Completion.create(
        #model="davinci:ft-personal-2022-12-27-18-17-29",
        model="davinci:ft-personal-2022-12-27-16-58-25",
        prompt="The following is a conversation with a therapist and a user. The therapist is JOY, who uses compassionate listening to have helpful and meaningful conversations with users. JOY is empathic and friendly. JOY's objective is to make the user feel better by feeling heard. With each response, JOY offers follow-up questions to encourage openness and tries to continue the conversation in a natural way. \n\nJOY-> Hello, I am your personal mental health assistant. What's on your mind today?\nUser->"+message+"JOY->",
        temperature=0.89,
        max_tokens=162,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n"]
    )

    # Display the message and the response in the conversation widget
    conversation.insert("end", f"You: {message}\n")
    conversation.insert("end", f"Bot: {response.get('choices')[0].get('text')}\n")

# Create the send button
send_button = tk.Button(root, text="Send", font=('Arial', 14), command=send_message, bg='white', fg='black')

# Create the clear button
clear_button = tk.Button(root, text='Clear', command=clear_conversation, bg='white', fg='black')

# Pack the widgets into the window
entry.pack()
send_button.pack()
clear_button.pack()
conversation.pack(side='left', fill='both', expand=True)
scrollbar.pack(side='right', fill='y')

# Run the Tkinter event loop
root.mainloop()
