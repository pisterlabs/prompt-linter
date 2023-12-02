# import openai
# import pyttsx3
# import pygame
# import sys
# import pygame.freetype

# # Initialize the Pygame library for the GUI
# pygame.init()
# pygame.freetype.init()

# # Set up the GUI window
# window_width = 800
# window_height = 600
# window = pygame.display.set_mode((window_width, window_height))
# pygame.display.set_caption("AI Girlfriend Chat")

# # Initialize pyttsx3 for text-to-speech
# engine = pyttsx3.init()

# # Set the voice properties (optional)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)  # You can change '0' to another index if you prefer a different voice

# # Initialize the OpenAI API
# openai.api_key = "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"

# # Initialize the conversation messages
# messages = [
#     {
#         "role": "system",
#         "content": "act as an extremely caring girlfriend"
#     },
#     {
#         "role": "user",
#         "content": "You mean so much to me, and I really appreciate your caring nature."
#     }
# ]

# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()

#     user_input = input("You: ")

#     # Add the user's input to the conversation
#     messages.append({
#         "role": "user",
#         "content": user_input
#     })

#     # Generate a response from OpenAI
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         temperature=0.2,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )

#     # Access and print the generated response
#     generated_text = response["choices"][0]["message"]["content"]
#     print(f"AI: {generated_text}")

#     # Say the generated response using pyttsx3
#     engine.say(generated_text)
#     engine.runAndWait()

#     # Clear the GUI window
#     window.fill((255, 255, 255))

#     # Render and display the conversation on the GUI
#     font = pygame.freetype.Font(None, 24)
#     y_offset = 50
#     for message in messages:
#         if message["role"] == "user":
#             color = (0, 0, 255)  # User's messages in blue
#         else:
#             color = (255, 0, 0)  # AI's messages in red

#         rendered_text, rect = font.render(message["content"], color)
#         window.blit(rendered_text, (10, y_offset))
#         y_offset += rect.height + 10

#     pygame.display.flip()
import openai
import pyttsx3
import tkinter as tk

# Initialize the OpenAI API
openai.api_key = "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"

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Create the Tkinter window
window = tk.Tk()
window.title("AI Girlfriend Chat")

# Create a text widget for displaying the conversation
conversation_text = tk.Text(window, height=10, width=40)
conversation_text.pack()

# Create an entry widget for user input
user_input_entry = tk.Entry(window, width=40)
user_input_entry.pack()

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

# Create a button to send user input
send_button = tk.Button(window, text="Send", command=send_user_input)
send_button.pack()

# Start the Tkinter main loop
window.mainloop()

