from datetime import timedelta
import os
import subprocess
import whisper
import tempfile
import argparse
import langchain
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from tqdm import tqdm

def get_translate_chain(from_lang, to_lang):
    template=f"You are a helpful assistant that translates {from_lang} to {to_lang}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="Please translate \"{text}\""+f" from {from_lang} to {to_lang}. Give me the translated {to_lang} directly without saying anything else, do not use \"."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # get a chat completion from the formatted messages
    chat = ChatOpenAI()
    chain = LLMChain(llm=chat, prompt=chat_prompt, verbose=True)
    return chain


def gen_srt(video_path, model_name="medium", from_language="English", to_language="Chinese", embed=False, translate=True):
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. use ffmpeg to extract audio from video and save it to Temp folder
        # Path to the temporary audio file
        temp_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        
        # Use ffmpeg to extract audio from video
        print("Extracting audio from video...")
        command = f"ffmpeg -i {video_path} -vn -ar 44100 -ac 2 -b:a 192k {temp_audio_path}"
        
        # Execute the command
        subprocess.call(command, shell=True)

        model = whisper.load_model(model_name)
        transcribe = model.transcribe(audio=temp_audio_path, language=from_language)
        segments = transcribe['segments']
        
        # 2. Use whisper to transcribe audio and save segments to srt file
        if translate:
            with get_openai_callback() as cb:
                chain = get_translate_chain(from_language, to_language)
                for segment in tqdm(segments):
                    segment['text'] = chain(segment['text'])['text']
                print(cb)
        
        # 3. Generate the SRT file
        srtFilename = video_path.split(".")[0] + ".srt"
        # overwrite the file if it already exists
        if os.path.exists(srtFilename):
            os.remove(srtFilename)
        for segment in segments:
            startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
            endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
            text = segment['text']
            segmentId = segment['id']+1
            segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"

            with open(srtFilename, 'a', encoding='utf-8') as srtFile:
                srtFile.write(segment)
        
        # 4. Use FFMPEG to embed srt file into video
        if not embed:
            return
        output_filename = video_path.split(".")[0] + "_subtitled.mp4"
        if os.path.exists(output_filename):
            os.remove(output_filename)
        embed_command = f"ffmpeg -i {video_path} -vf subtitles={srtFilename} {output_filename}"
        subprocess.call(embed_command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments')

    # Add the arguments
    parser.add_argument('-i', type=str, required=True, dest='input_file',
                        help='Input file name')

    parser.add_argument('-m', type=str, default='medium', dest='model_name',
                        help='Model type, default is "medium"')

    parser.add_argument('-f', type=str, default='English', dest='from_lang',
                        help='Translate from language, default is "English"')

    parser.add_argument('-t', type=str, default='Chinese', dest='to_lang',
                        help='Translate to language, default is "Chinese"')

    parser.add_argument('--embed', dest='embed', action='store_true',
                        help='Whether to Embed subtitles, default is False')

    parser.add_argument('--translate', dest='translate', action='store_true',
                        help='Whether to Translate, default is False')
    args = parser.parse_args()

    gen_srt(args.input_file, model_name=args.model_name, embed=args.embed, translate=args.translate, from_language=args.from_lang, to_language=args.to_lang)
