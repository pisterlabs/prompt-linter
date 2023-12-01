import os
import tkinter as tk
import tkinter.messagebox as messagebox
import logging
import openai

logging.basicConfig(filename='translator.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

class SlangTranslatorApp:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        openai.api_key = self.openai_api_key

        self.main_window = tk.Tk()
        self.main_window.title("Slang Translator")

        self.setup_labels()
        self.setup_text_widgets()
        self.setup_translate_button()
        self.pack_gui_elements()

    def run(self):
        self.main_window.mainloop()

    def setup_labels(self):
        self.input_label = tk.Label(self.main_window, text="Enter slang or vernacular text:")
        self.context_label = tk.Label(self.main_window, text="Enter context (optional):")
        self.output_label = tk.Label(self.main_window, text="Standard English translation:")

    def setup_text_widgets(self):
        self.input_text_widget = self.create_text_widget()
        self.context_text_widget = self.create_text_widget()
        self.output_text_widget = self.create_text_widget()

    def create_text_widget(self):
        return tk.Text(self.main_window, height=10)

    def setup_translate_button(self):
        self.translate_button = tk.Button(self.main_window, text="Translate", command=self.handle_translate_button_click)

    def pack_gui_elements(self):
        self.input_label.pack()
        self.input_text_widget.pack()
        self.context_label.pack()
        self.context_text_widget.pack()
        self.output_label.pack()
        self.output_text_widget.pack()
        self.translate_button.pack()

    def handle_translate_button_click(self):
        user_input = self.input_text_widget.get("1.0", "end-1c").strip()
        context_input = self.context_text_widget.get("1.0", "end-1c").strip()

        if not user_input:
            messagebox.showerror("Input Error", "Input cannot be empty. Please enter a valid message.")
            return

        conversation = self.create_conversation_object(user_input, context_input)
        self.call_openai_api(conversation)

    def create_conversation_object(self, user_input, context=None):
        system_message = self.create_conversation_message("system", "ou are a specialized AI assistant that understands and translates current internet slang, online lingo, vernacular language, colloquialisms, and regional dialects into standard English. Your task is to translate these nuanced forms of communication accurately, taking into account their evolving nature, cultural context, and how they're used in informal online conversations. When a user gives you a slang phrase or term, translate it, explain its meaning, and provide context about where and how it's typically used. Please on the first line of your response, write 'The translation is:' which will output a standard plain english translation. On the second line, write 'The context is:' which will output the context of the slang term.")
        user_message = self.create_conversation_message("user", user_input)

        messages = [system_message, user_message]
        if context:
            context_message = self.create_conversation_message("user", f"The context is: {context}")
            messages.append(context_message)

        return messages

    @staticmethod
    def create_conversation_message(role, content):
        return {"role": role, "content": content}

    def call_openai_api(self, conversation):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=conversation,
                temperature=0.5,
                max_tokens=500
            )
            response_text = response.choices[0].message['content']
            self.output_text_widget.delete("1.0", "end")
            self.output_text_widget.insert("end", response_text)
        except Exception as e:
            logging.error(f"An error occurred while using OpenAI API: {str(e)}")
            messagebox.showerror("Error", "An error occurred while processing your request.")

if __name__ == "__main__":
    app = SlangTranslatorApp()
    app.run()
