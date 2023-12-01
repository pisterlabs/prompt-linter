import tkinter as tk
from tkinter import Scrollbar, Text, Entry, Button
import threading
import openai
import pyperclip
import functions
import sv_ttk
import tiktoken


class ChatGUI:
	def __init__(self, root):
		self.root = root
		self.root.title("VecchioGPT Chat")
		# Get encoder for calculating tokens in each request
		self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
		# self.root.geometry("800x600")

		entry_font = ("Montserrat", 13)
		self.chat_text_history = ""
		self.chat_history = Text(root, wrap="word", state="disabled", font = entry_font)

		# Create the chat input box with indentation and without border
		self.chat_input = Entry(root, width=50, bd=0, insertwidth=4, font = entry_font)
		self.send_button = Button(root, text="Send", command=self.send_message, font=entry_font)

		self.chat_history.pack(expand=True, fill="both")
		self.chat_input.pack(side="left", expand=True, fill="x")
		self.send_button.pack(side="right")
		sv_ttk.set_theme("dark")

		self.append_message("ChatGPT: Hello! Ask me anything.")

		self.chat_input.bind("<Return>", self.on_enter_pressed)
		self.chat_input.bind("<Control-Return>", self.insert_newline)
		self.chat_input.bind("<Shift-Return>", self.insert_newline)
		
		# Set focus on the chat input box when the GUI is opened
		self.chat_input.focus_set()
		# Additional member variables
		self.pending_responses = []

		# Start a timer to periodically check for new responses
		self.root.after(100, self.check_responses)

		self.show_window()

	def update_message_history(self, message, type):
		self.chat_text_history += f"Previous {type} message:\n" + message + "\n\n"

	def append_message(self, message):
		self.chat_history.config(state="normal")
		self.chat_history.insert("end", message + "\n\n")
		self.chat_history.config(state="disabled")
		self.chat_history.see("end")


	def send_message(self):
		user_input = self.chat_input.get()
		if user_input:
			self.append_message(f"You: {user_input}")
			self.chat_input.delete(0, "end")
			t = threading.Thread(target=lambda: self.get_chatgpt_response(user_input))
			t.start()
			# Start checking periodically if the thread has finished.


	def get_chatgpt_response(self, user_input):
			try:
				tokens_in_input = len(self.encoder.encode(user_input))
				available_other_tokens = 4096 - tokens_in_input
				tokenized_chat_text_history = self.encoder.encode(self.chat_text_history)
				if len(tokenized_chat_text_history) > available_other_tokens:
					self.chat_text_history = self.encoder.decode(tokenized_chat_text_history[-available_other_tokens:])
				truncated_chat_text_history = self.chat_text_history

				input_with_previous_messages = "Current request:" + user_input + "\n\n" + truncated_chat_text_history
				response = openai.ChatCompletion.create(
					model="gpt-3.5-turbo",

					messages=[
						{"role": "system", "content": "Continue the conversation below. Pay special attention to the current request."},
						{"role": "user", "content": input_with_previous_messages},
					],
				)

				chatgpt_reply = response.choices[0].message["content"]

				self.update_message_history(user_input, "User")
				self.update_message_history(chatgpt_reply, "ChatGPT")
				self.append_message(f"ChatGPT: {chatgpt_reply}")
				pyperclip.copy(chatgpt_reply)  # Copy to clipboard

			except Exception as e:
				print(f"Error getting ChatGPT response: {e}")


	def on_enter_pressed(self, event):
		self.send_message()

	def insert_newline(self, event):
		self.chat_input.insert(tk.END, '\n')

	def check_responses(self):
		# Check if there are pending responses
		if self.pending_responses:
			response = self.pending_responses.pop(0)
			self.append_message(f"ChatGPT: {response}")
			pyperclip.copy(response)  # Copy to clipboard

		# Schedule the method to run again after 100 milliseconds
		self.root.after(100, self.check_responses)

	def show_window(self):
		"""
		Shows the GUI window.
		"""
		self.bring_to_front()
		
	def bring_to_front(self):
		"""
		Brings the GUI window to the front.
		"""
		self.root.lift()
		self.root.attributes('-topmost', True)
		self.root.after_idle(self.root.attributes, '-topmost', False)

if __name__ == "__main__":
	root = tk.Tk()
	chat_gui = ChatGUI(root)

	def start_gui():
		root.mainloop()

	start_gui().start()
