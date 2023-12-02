from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.core import window


import openai

class ChatApplication(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, 1)

        # NOTE: hierarchy works from top to bottom
        # A widget that follows another widget in the code will overlay the previous one

        # Change window color
        window.Window.clearcolor = (0.5, 0.5, 0.5, 1)

        # Create the image layout
        self.image_layout = FloatLayout(size_hint=(1, 1))
        self.add_widget(self.image_layout)

        # Create the text bubble image
        self.bubble = Image(source='assets/bubble.png', size_hint=(None, None),
                            pos_hint={'center_x': 0.5, 'center_y': 0.45}, height=Window.height * 0.3, allow_stretch=True, keep_ratio=False,
                            width=Window.width * 0.35)
        self.image_layout.add_widget(self.bubble)

        # Create the text bubble layout
        self.text_bubble_layout = FloatLayout(size_hint=(None, None), height=Window.height*0.3, width=Window.width*0.35,
                                              pos_hint={'center_x': 0.5, 'center_y': 0.45})
        self.image_layout.add_widget(self.text_bubble_layout)

        # Create the image
        self.image = Image(source='assets/lizz_body.png', size_hint=(1, 1), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.image_layout.add_widget(self.image)

        # Create the text bubble
        self.text_bubble = Label(text='', size_hint=(1, 1), pos_hint={'center_x': 0.5, 'center_y': 0.385},
                                 text_size=(Window.width * 0.3, None), bold=True)
        self.text_bubble_layout.add_widget(self.text_bubble)
        self.text_bubble.font_size = 28
        self.text_bubble.color = (0, 0, 0, 1)  # set the text color to black

        # Create the text input
        self.input_frame = RelativeLayout(size_hint=(None, None), width=Window.width * 0.35, height=Window.height*0.075, pos_hint={'center_x': 0.5, 'center_y': 0.175})
        self.image_layout.add_widget(self.input_frame)

        self.input_entry = TextInput(multiline=True, font_size=28, size=(Window.width, 1),
                                     pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.input_frame.add_widget(self.input_entry)

        self.input_button = Button(text='Send', on_press=self.send_message, size_hint=(0.15, 1),
                                   pos_hint={'right': 1, 'center_y': 0.5})
        self.input_frame.add_widget(self.input_button)

        # Initialize OpenAI API credentials
        API_KEY = "INSERT API KEY"
        openai.api_key = API_KEY

        # Initialize conversation history with initial system prompt
        self.messages = [{"role": "system", "content": "You are a social companion chatbot for seniors called Lizz. Have a conversation with me. Keep the conversation flowing. Keep answers under 40 words. Speak dutch with me."}]

    def send_message(self, instance):
        # Get the message from the input box
        message = self.input_entry.text

        # Add the user message to the conversation history
        self.messages.append({"role": "user", "content": message})

        # Call the OpenAI API and get the response
        response = self.interact_with_chatGPT(self.messages)
        if response:
            # If the response was successfully retrieved, display it in the text bubble
            self.text_bubble.text = response
            # Add the assistant message to the conversation history
            self.messages.append({"role": "assistant", "content": response})
        else:
            # If the response was not successfully retrieved, display an error message
            self.text_bubble.text = 'Error retrieving response from API'

        # Clear the input box
        self.input_entry.text = ''

    def interact_with_chatGPT(self, messages):
        try:
            # Call the OpenAI API to get the response
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages, max_tokens=55)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error interacting with OpenAI API: {e}")
            return None

class ChatApplicationApp(App):
    def build(self):
        return ChatApplication()

if __name__ == '__main__':
    ChatApplicationApp().run()