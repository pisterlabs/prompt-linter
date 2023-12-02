import openai
import pyttsx3
import tkinter as tk

# Initialize the OpenAI API
openai.api_key = "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"

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Create the Tkinter window
window = tk.Tk()
window.title("AI Girlfriend Chat")

# Create a frame for conversation display
conversation_frame = tk.Frame(window)
conversation_frame.pack(pady=10)

# Create a text widget for displaying the conversation with custom styling
conversation_text = tk.Text(
    conversation_frame,
    height=15,
    width=50,
    wrap=tk.WORD,
    borderwidth=0,
    padx=10,
    pady=10,
    font=("Arial", 12),
)
conversation_text.pack(side=tk.LEFT)

# Create a scrollbar for the conversation text widget
conversation_scrollbar = tk.Scrollbar(conversation_frame)
conversation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
conversation_text.config(yscrollcommand=conversation_scrollbar.set)
conversation_scrollbar.config(command=conversation_text.yview)

# Create an entry widget for user input with styling
user_input_entry = tk.Entry(
    window,
    width=50,
    font=("Arial", 12),
)
user_input_entry.pack(pady=10)

# Create a button to send user input with custom styling
send_button = tk.Button(
    window,
    text="Send",
    command=send_user_input,
    width=10,
    height=2,
    font=("Arial", 12),
    bg="lightblue",
    activebackground="lightblue",
)
send_button.pack()

# Function to send user input and receive AI responses
def send_user_input():
    user_input = user_input_entry.get()

    # Add the user's input to the conversation
    conversation_text.insert(tk.END, f"You: {user_input}\n")
    conversation_text.update_idletasks()

    # Generate a response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "act as an extremely caring girlfriend"
            },
            {
                "role": "user",
                "content": "You mean so much to me, and I really appreciate your caring nature."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.2,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Access and print the generated response
    generated_text = response["choices"][0]["message"]["content"]
    conversation_text.insert(tk.END, f"AI: {generated_text}\n")
    conversation_text.update_idletasks()

    # Say the generated response using pyttsx3
    engine.say(generated_text)
    engine.runAndWait()

    # Clear the user input entry
    user_input_entry.delete(0, tk.END)

# Start the Tkinter main loop
window.mainloop()
# Function to send user input and receive AI responses
# def send_user_input():
#     user_input = user_input_entry.get()

#     # Add the user's input to the conversation
#     conversation_text.insert(tk.END, f"You: {user_input}\n")
#     conversation_text.update_idletasks()

#     # Generate a response from OpenAI
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "act as an extremely caring girlfriend"
#             },
#             {
#                 "role": "user",
#                 "content": "You mean so much to me, and I really appreciate your caring nature."
#             },
#             {
#                 "role": "user",
#                 "content": user_input
#             }
#         ],
#         temperature=0.2,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )

#     # Access and print the generated response
#     generated_text = response["choices"][0]["message"]["content"]
#     conversation_text.insert(tk.END, f"AI: {generated_text}\n")
#     conversation_text.update_idletasks()

#     # Clear the user input entry
#     user_input_entry.delete(0, tk.END)

# # Start the Tkinter main loop
# window.mainloop()