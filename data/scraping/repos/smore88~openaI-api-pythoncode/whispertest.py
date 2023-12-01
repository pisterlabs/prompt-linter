#!/usr/bin/env python3 
import openai
import gradio as gr
from dotenv import load_dotenv
import os 
import whisper
from langchain.callbacks import get_openai_callback

#load_dotenv()

model = whisper.load_model("small")

class whiperExample(object):
    def __init__(self, ui_obj) -> None:
        self.name = "Sample Audio App"
        self.description = "audio -> text"
        self.api_key = None
        self.api_key_status = "API Key not found"
        self.ui_obj = ui_obj
    
    def createUI(self):
        with self.ui_obj:
            gr.Markdown(self.name)
            with gr.Tabs():
                with gr.TabItem("Set up Key"):
                    with gr.Row():
                        openai_api_key = gr.Textbox(
                            label="OpenAI API Key",
                            placeholder="add ur key",
                            type="password"
                        )
                        set_api_action = gr.Button("Set up Key")
                    with gr.Row():
                        openai_api_key_status = gr.Label(self.api_key_status)
                with gr.TabItem("set up audio"):
                    with gr.Row():
                        my_sample_audio = gr.components.Audio(source="microphone", type="filepath")
                        set_transcribe_action = gr.Button("transcribe")
                    with gr.Row():
                        transcribe_output = gr.Label("Transcription")

            set_api_action.click(
                self.update_api_status,[
                    openai_api_key
                ], [
                    openai_api_key_status
                ]
            )

            set_transcribe_action.click(
                self.transcribe,[
                    my_sample_audio
                ], [
                    transcribe_output
                ]
            )



    def transcribe(self, audio):
        #loads audio file
        audio = whisper.load_audio(audio)
        #trims audio file
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        print(f"Detence language: {max(probs, key=probs.get)}")

        #decode it
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        return result.text
    
    def update_api_status(self, api_key):
        if api_key is not None and len(api_key) > 0:
            self.api_key = str(api_key)
            self.api_key_status = f"Found API Key"
            os.environ["OPENAI_API_KEY"] = self.api_key
        return self.api_key_status
    
    def launchUI(self):
        self.ui_obj.launch()

if __name__ == '__main__':
    openai.api_key = os.getenv("OPEN_API_KEY")
    myapp = gr.Blocks()
    myapp_ui = whiperExample(myapp)
    myapp_ui.createUI()
    myapp_ui.launchUI()


    
