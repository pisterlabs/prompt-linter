import gradio as gr
from typing_extensions import Literal

from openai_client import *
from openai_vision import *
from openai_assistant import *


class UI:
    def __init__(self, assistant: Assistant, client: OpenAI):
        self.assistant = assistant
        self.client = client
        self.AVATARS = (
            "https://media.roboflow.com/spaces/roboflow_raccoon_full.png",
            "https://media.roboflow.com/spaces/openai-white-logomark.png",
        )

    def chat_tab(self):
        gr.Markdown("# <center> Chat </center>")
        self.chat_chatbot = gr.Chatbot(height=500, bubble_full_width=False, avatar_images=self.AVATARS)
        with gr.Row():
            self.chat_textbox = gr.Textbox(label="Chatbox", placeholder="Type your message here", scale=3)
            self.chat_model = gr.Dropdown(choices=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-1106-preview"], label="Model", value="gpt-3.5-turbo")
        with gr.Row():
            self.chat_audio_input = gr.Audio(label="Audio Input", source="microphone", type="filepath")
            self.chat_audio_output = gr.Audio(label="Audio Output", autoplay=True)
        self.chat_clear_button = gr.ClearButton([self.chat_textbox, self.chat_chatbot])

    def chat_tab_callbacks(self):
        self.chat_textbox.submit(
            fn=self.chatbot_response,
            inputs=[self.chat_textbox, self.chat_chatbot, self.chat_model],
            outputs=[self.chat_textbox, self.chat_chatbot],
        )

        self.chat_audio_input.stop_recording(
            fn=self.ui_speech_to_text,
            inputs=[self.chat_audio_input, self.stt_model, self.stt_response_type],
            outputs=[self.chat_textbox],
        ).then(
            fn=self.chatbot_response,
            inputs=[self.chat_textbox, self.chat_chatbot, self.chat_model],
            outputs=[self.chat_textbox, self.chat_chatbot],
        ).then(
            self.chatbot_text_to_speech,
            inputs=[self.chat_chatbot, self.tts_model, self.tts_voice, self.tts_output_file_format, self.tts_speed],
            outputs=[self.chat_audio_output],
        )

    def chat_with_vision_tab(self):
        gr.Markdown("# <center> Chat with Vision </center>")
        with gr.Row():
            self.vision_webcam = gr.Image(source="webcam", streaming=True)  # Fix uploading flicker
            with gr.Column():
                self.vision_chatbot = gr.Chatbot(height=500, bubble_full_width=False, avatar_images=self.AVATARS)
                self.vision_textbox = gr.Textbox(label="Chatbox", placeholder="Type your message here")
                with gr.Row():
                    self.vision_audio_input = gr.Audio(label="Audio Input", source="microphone", type="filepath")
                    self.vision_audio_output = gr.Audio(label="Audio Output", autoplay=True)
                self.vision_clear_button = gr.ClearButton([self.vision_textbox, self.vision_chatbot])

    def chat_with_vision_tab_callbacks(self):
        self.vision_textbox.submit(
            fn=self.ui_respond_with_vision,
            inputs=[self.vision_webcam, self.vision_textbox, self.vision_chatbot],
            outputs=[self.vision_textbox, self.vision_chatbot],
        )

        self.vision_audio_input.stop_recording(
            fn=self.ui_speech_to_text,
            inputs=[self.vision_audio_input, self.stt_model, self.stt_response_type],
            outputs=[self.vision_textbox],
        ).then(
            fn=self.ui_respond_with_vision,
            inputs=[self.vision_webcam, self.vision_textbox, self.vision_chatbot],
            outputs=[self.vision_textbox, self.vision_chatbot],
        ).then(
            fn=self.chatbot_text_to_speech,
            inputs=[self.vision_chatbot, self.tts_model, self.tts_voice, self.tts_output_file_format, self.tts_speed],
            outputs=[self.vision_audio_output],
        )

    def assistant_tab(self):
        gr.Markdown("# <center> Assistant </center>")
        self.assistant_chatbot = gr.Chatbot(height=500, bubble_full_width=False, avatar_images=self.AVATARS)
        with gr.Row():
            self.assistant_textbox = gr.Textbox(label="Chatbox", placeholder="Type your message here", scale=3)
            self.assistant_id = gr.Textbox(label="Assistant Id", value=self.assistant.assistant_id)
        with gr.Row():
            self.assistant_audio_input = gr.Audio(label="Audio Input", source="microphone", type="filepath")
            self.assistant_audio_output = gr.Audio(label="Audio Output", autoplay=True)
        self.assistant_clear_button = gr.ClearButton([self.assistant_textbox, self.assistant_chatbot])

    def assistant_tab_callbacks(self):
        self.assistant_id.change(
            fn=self.assistant.set_assistant,
            inputs=[self.assistant_id],
        )

        self.assistant_textbox.submit(
            fn=self.chatbot_assistant_respond,
            inputs=[self.assistant_textbox, self.assistant_chatbot, self.assistant_id],
            outputs=[self.assistant_textbox, self.assistant_chatbot],
        )

        self.assistant_audio_input.stop_recording(
            fn=self.ui_speech_to_text,
            inputs=[self.assistant_audio_input, self.stt_model, self.stt_response_type],
            outputs=[self.assistant_textbox],
        ).then(
            fn=self.chatbot_assistant_respond,
            inputs=[self.assistant_textbox, self.assistant_chatbot, self.assistant_id],
            outputs=[self.assistant_textbox, self.assistant_chatbot],
        ).then(
            fn=self.chatbot_text_to_speech,
            inputs=[self.assistant_chatbot, self.tts_model, self.tts_voice, self.tts_output_file_format, self.tts_speed],
            outputs=[self.assistant_audio_output],
        )

    def text_to_speech_tab(self):
        gr.Markdown("# <center> Text to Speech </center>")
        with gr.Row(variant="panel"):
            self.tts_model = gr.Dropdown(choices=["tts-1", "tts-1-hd"], label="Model", value="tts-1-hd")
            self.tts_voice = gr.Dropdown(choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"], label="Voice Options", value="alloy")
            self.tts_output_file_format = gr.Dropdown(choices=["mp3", "opus", "aac", "flac"], label="Output Options", value="mp3")
            self.tts_speed = gr.Slider(minimum=0.25, maximum=4.0, value=1.0, step=0.01, label="Speed")
        self.tts_textbox = gr.Textbox(label="Input text", placeholder='Enter your text and then click on the "Text to Speech" button, or press the Enter key.')
        self.tts_button = gr.Button("Text to Speech")
        self.tts_output_audio = gr.Audio(label="Speech Output", autoplay=True)

    def text_to_speech_tab_callbacks(self):
        self.tts_button.click(
            fn=self.ui_text_to_speech,
            inputs=[self.tts_textbox, self.tts_model, self.tts_voice, self.tts_output_file_format, self.tts_speed],
            outputs=self.tts_output_audio,
        )

        self.tts_textbox.submit(
            fn=self.ui_text_to_speech,
            inputs=[self.tts_textbox, self.tts_model, self.tts_voice, self.tts_output_file_format, self.tts_speed],
            outputs=self.tts_output_audio,
        )

    def speech_to_text_tab(self):
        gr.Markdown("# <center> Speech to Text </center>")
        with gr.Row(variant="panel"):
            self.stt_model = gr.Dropdown(choices=["whisper-1"], label="Model", value="whisper-1")
            self.stt_response_type = gr.Dropdown(choices=["json", "text", "srt", "verbose_json", "vtt"], label="Response Type", value="text")
        with gr.Row():
            self.stt_audio_input = gr.Audio(source="microphone", type="filepath")
            self.stt_file = gr.UploadButton(file_types=[".mp3", ".wav"], label="Select File", type="filepath")
        self.stt_output_text = gr.Text(label="Output Text")

    def speech_to_text_tab_callbacks(self):
        self.stt_audio_input.stop_recording(
            fn=self.ui_speech_to_text,
            inputs=[self.stt_audio_input, self.stt_model, self.stt_response_type],
            outputs=self.stt_output_text,
        )

        self.stt_file.upload(fn=self.ui_speech_to_text, inputs=[self.stt_file, self.stt_model, self.stt_response_type], outputs=self.stt_output_text)

    def image_generation_tab(self):
        gr.Markdown("# <center> Image Generation </center>")
        with gr.Row(variant="panel"):
            self.image_model = gr.Dropdown(choices=["dall-e-2", "dall-e-3"], label="Model", value="dall-e-3")
            self.image_quality = gr.Dropdown(choices=["standard", "hd"], label="Quality", value="standard")
            self.image_size = gr.Dropdown(choices=["1024x1024", "1792x1024", "1024x1792"], label="Size", value="1024x1024")
            self.image_style = gr.Dropdown(choices=["vivid", "natural"], label="Style", value="vivid")

        self.image_textbox = gr.Textbox(
            label="Input Text", placeholder='Enter your text and then click on the "Image Generate" button, or press the Enter key.'
        )
        self.image_button = gr.Button("Image Generate")
        self.image_output_image = gr.Image(label="Image Output")

    def image_generation_tab_callbacks(self):
        self.image_textbox.submit(
            fn=self.ui_generate_image,
            inputs=[self.image_textbox, self.image_model, self.image_quality, self.image_size, self.image_style],
            outputs=self.image_output_image,
        )

        self.image_button.click(
            fn=self.ui_generate_image,
            inputs=[self.image_textbox, self.image_model, self.image_quality, self.image_size, self.image_style],
            outputs=self.image_output_image,
        )

    # ui specific functions

    def ui_speech_to_text(self, audio, model, response_type):
        text = speech_to_text(self.client, audio, model, response_type)
        return text

    def ui_text_to_speech(self, text, model, voice, output_file_format, speed):
        audio_path = text_to_speech(self.client, text, model, voice, output_file_format, speed)
        return audio_path

    def ui_respond_with_vision(self, image: np.ndarray, prompt: str, chat_history):
        openai_key = self.client.api_key
        image = preprocess_image(image=image)
        image_path = cache_image(image)
        response = prompt_image(api_key=openai_key, image=image, prompt=prompt)
        chat_history.append(((image_path, None), None))
        chat_history.append((prompt, response))
        return "", chat_history

    def ui_generate_image(self, prompt, model, quality, size, style):
        image_url = generate_image(self.client, prompt, model, quality, size, style)
        return image_url

    def chatbot_response(self, prompt, chat_history, model):
        response = chat(self.client, prompt, model)
        chat_history.append((prompt, response))
        return "", chat_history

    def chatbot_text_to_speech(self, chat_history, model, voice, output_file_format, speed):
        text = chat_history[-1][-1]
        audio = text_to_speech(self.client, text, model, voice, output_file_format, speed)
        return audio

    def chatbot_assistant_respond(self, prompt, chat_history, assistant_id):
        self.assistant.set_assistant(assistant_id)
        self.assistant.create_message(prompt)
        self.assistant.create_run()
        if self.assistant.wait_for_run():
            message = self.assistant.get_last_message()
            chat_history.append((prompt, message))
        return "", chat_history

    # create and launch ui

    def launch(self):
        with gr.Blocks() as gradio:
            with gr.Tab(label="Chat"):
                self.chat_tab()
            with gr.Tab(label="Chat with Vision"):
                self.chat_with_vision_tab()
            with gr.Tab(label="Assistant"):
                self.assistant_tab()
            with gr.Tab(label="Text to Speech"):
                self.text_to_speech_tab()
            with gr.Tab(label="Speech to Text"):
                self.speech_to_text_tab()
            with gr.Tab(label="Image Generation"):
                self.image_generation_tab()

            self.chat_tab_callbacks()
            self.chat_with_vision_tab_callbacks()
            self.assistant_tab_callbacks()
            self.text_to_speech_tab_callbacks()
            self.speech_to_text_tab_callbacks()
            self.image_generation_tab_callbacks()

        gradio.launch()
