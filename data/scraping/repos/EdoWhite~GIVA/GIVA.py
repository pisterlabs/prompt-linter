from transformers import WhisperProcessor, WhisperForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import gradio as gr
import librosa
import openai
import numpy as np

def load_processors_asr(asr_model):
    # load model and processor for ASR
    checkpoint_asr = asr_model
    processor_asr = WhisperProcessor.from_pretrained(checkpoint_asr)
    model_asr = WhisperForConditionalGeneration.from_pretrained(checkpoint_asr)
    model_asr.config.forced_decoder_ids = None
    return processor_asr, model_asr


def load_processor_tts(tts_model, tts_vocoder):
    # load model and processor for TTS
    checkpoint_tts = tts_model
    processor_tts = SpeechT5Processor.from_pretrained(checkpoint_tts)
    model_tts = SpeechT5ForTextToSpeech.from_pretrained(checkpoint_tts)
    vocoder_tts = SpeechT5HifiGan.from_pretrained(tts_vocoder)
    return processor_tts, model_tts, vocoder_tts


def process_audio(sampling_rate, waveform):
    # convert to float
    waveform = waveform / 32678.0
    # convert to mono
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform.T)
    # resample to 16 kHz
    if sampling_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=16000)
    # make array
    waveform = np.array(waveform)
    return waveform


def transcript(asr_model, audio):
    if audio is not None:
        sampling_rate, waveform = audio
    else:
        raise gr.Error("Start the recording!")
    
    processor_asr, model_asr = load_processors_asr(asr_model)

    waveform = process_audio(sampling_rate, waveform)
    
    input = processor_asr(audio=waveform, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model_asr.generate(input)
    transcription = processor_asr.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]


def textToSpeech(tts_model, tts_vocoder, text):
    if len(text.strip()) == 0:
        return (16000, np.zeros(0).astype(np.int16))
    
    processor_tts, model_tts, vocoder_tts = load_processor_tts(tts_model, tts_vocoder)

    speaker_embedding = np.load("speaker_embeddings/cmu_us_clb_arctic-wav-arctic_a0144.npy")
    speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0)

    inputs = processor_tts(text=text, return_tensors="pt")
    speech = model_tts.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder_tts)
    speech = (speech.numpy() * 32767).astype(np.int16)
    return (16000, speech)


def get_completion(prompt, model, temperature, max_tokens, presence_penality, frequency_penality):
    messages = [
        {"role": "system",
         "content": "you are a voice assistant. Keep the answer short and concise, please. Also consider that your output will be converted into audio, so make sure to provide a text that makes sense even if listened."
         },
        {"role": "user", 
         "content": prompt
         }
        ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        presence_penality=presence_penality,
        frequency_penality=frequency_penality
    )
    return response.choices[0].message["content"]


def chat(asr_model, tts_model, tts_vocoder, gpt_model, temperature, max_tokens, presence_penality, frequency_penality, openAI_key, audio):
    if openAI_key != "":
        openai.api_key = openAI_key
    else:
        raise gr.Error("Provide a valid OpenAI API Key!")

    try:
        # Automatic Speech Recognition
        prompt = transcript(asr_model, audio)
        # Get GPT Completion
        generated_text = get_completion(prompt, gpt_model, temperature, max_tokens, presence_penality, frequency_penality)
        # Text to Speech
        answer = textToSpeech(tts_model, tts_vocoder, generated_text)
        return prompt, generated_text, answer
    except:
        raise gr.Error("Check the recording, the API key, or the parameters")


title="GIVA - GPT-based Vocal Virtual Assistant"

description = """
GIVA is a vocal assistant that combines speech recognition and text-to-speech with the capabilities of GPT (3.5-turbo or 4). Prompts are engineered so that GPT provides outputs that are short and adapted to be converted to audio.

### Features:
- **Speech Recognition**: GIVA employs the `openai/whisper` model for accurate transcription of speech inputs. It's possibile to choose between the tiny, small, medium, and large v2 versions of the mdoel.
- **GPT Chat Completion**: The user can choose between GPT-3.5-turbo and GPT-4 to interact with the vocal assistant.
- **Text-to-Speech**: With the `microsoft/speecht5_tts` model, GIVA generates an audio output.
- **Interactive Interface**: The application consists of two tabs. The first tab exclusively presents the audio output, while the second tab provides additional information, including the output of Automatic Speech Recognition (ASR) and the responses generated by GPT.

### ASR Models:
The user can select from different ASR models, such as:
- [OpenAI Whisper-tiny](https://huggingface.co/openai/whisper-tiny)
- [OpenAI Whisper-small](https://huggingface.co/openai/whisper-small)
- [OpenAI Whisper-medium](https://huggingface.co/openai/whisper-medium)
- [OpenAI Whisper-large-v2](https://huggingface.co/openai/whisper-large-v2)

### GPT Models
The user can select from different ASR models, such as:
- GPT-3.5-turbo
- GPT-4

### References:
- [OpenAI Whisper-base](https://huggingface.co/openai/whisper-tiny)
- [Microsoft SpeechT5_tts](https://huggingface.co/microsoft/speecht5_tts)
- [Matthijs, Huggingface - Speech Synthesis, Recognition, and More With SpeechT5](https://huggingface.co/blog/speecht5)
- [Huggingface - ASR with Transformers](https://huggingface.co/docs/transformers/tasks/asr)
- [OpenAI API Reference](https://platform.openai.com)

"""

theme = gr.themes.Soft(
    primary_hue="lime",
    secondary_hue="fuchsia",
    neutral_hue="zinc",
)

onlyAudioOutputTab = gr.Interface(
    theme=theme,
    fn=chat,
    inputs=[
        gr.Dropdown(["openai/whisper-tiny", "openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v2"], 
                    label="Select Speech Recognition Model Checkpoint",
                    value="openai/whisper-small"),

        gr.Dropdown(["microsoft/speecht5_tts"], 
                    label="Select Text-to-Speech Model Checkpoint", 
                    value="microsoft/speecht5_tts"),

        gr.Dropdown(["microsoft/speecht5_hifigan"], 
                    label="Select Vocoder Checkpoint", 
                    value="microsoft/speecht5_hifigan"),

        gr.Dropdown(["gpt-3.5-turbo", "gpt-4"], 
                    label="Select GPT Model", 
                    value="gpt-3.5-turbo"),
        
        gr.Slider(0, 2, value=1, label="Temperature", info="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."),
        gr.Number(1000, label="Max Tokens", info="The maximum number of tokens to generate in the chat completion."),
        gr.Slider(-2.0, 2.0, value=0, label="Presence Penality", info="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."),
        gr.Slider(-2.0, 2.0, value=0, label="Frequency Penality", info="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."),

        gr.Text(label="Provide an OpenAI API Key", type="password"),
        gr.Audio(label="Record", source="microphone", type="numpy")
    ],
    outputs=[
        gr.Text(label="Transcription", visible=False),
        gr.Text(label="GPT Answer", visible=False),
        gr.Audio(label="Speech Answer", type="numpy")
    ],
    description=description,
    allow_flagging="never"
)


AudioTextOutput = gr.Interface(
    theme=theme,
    fn=chat,
    inputs=[
        gr.Dropdown(["openai/whisper-tiny", "openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v2"], 
                    label="Select Speech Recognition Model Checkpoint",
                    value="openai/whisper-small"),

        gr.Dropdown(["microsoft/speecht5_tts"], 
                    label="Select Text-to-Speech Model Checkpoint", 
                    value="microsoft/speecht5_tts"),

        gr.Dropdown(["microsoft/speecht5_hifigan"], 
                    label="Select Vocoder Checkpoint", 
                    value="microsoft/speecht5_hifigan"),

        gr.Dropdown(["gpt-3.5-turbo", "gpt-4"], 
                    label="Select GPT Model", 
                    value="gpt-3.5-turbo"),
        
        gr.Slider(0, 2, value=1, label="Temperature", info="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."),
        gr.Number(1000, label="Max Tokens", info="The maximum number of tokens to generate in the chat completion."),
        gr.Slider(-2.0, 2.0, value=0, label="Presence Penality", info="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."),
        gr.Slider(-2.0, 2.0, value=0, label="Frequency Penality", info="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."),

        gr.Text(label="Provide an OpenAI API Key", type="password"),
        gr.Audio(label="Record", source="microphone", type="numpy")
    ],
    outputs=[
        gr.Text(label="Transcription"),
        gr.Text(label="GPT Completion"),
        gr.Audio(label="Speech Answer", type="numpy")
    ],
    description=description,
    allow_flagging="never"
)

demo = gr.TabbedInterface(
    interface_list = [onlyAudioOutputTab, AudioTextOutput], 
    tab_names = ["Audio Output", "Transcript + Completion + Audio Output"],
    title=title
).launch()