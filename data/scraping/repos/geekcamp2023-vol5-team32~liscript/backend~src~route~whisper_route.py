import os
import uuid
from pydub import AudioSegment

import openai
from dotenv import load_dotenv
from flask import Blueprint, jsonify, request
from utils.listener import callWhisper
from werkzeug.utils import secure_filename

whisper_module = Blueprint("whisper_route", __name__)

basedir = os.path.dirname(__file__)
load_dotenv(verbose=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_FOLDER = os.path.join(basedir, "../audio/")
ALLOWED_EXTENSIONS = {'mp3',"m4a","mp4","wav"}

MAX_AUDIO_SIZE = 26214400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def split_and_reduce_audio(source_audio, chunk_length_ms):
    chunks = []
    total_duration = len(source_audio)
    chunk_count = total_duration // chunk_length_ms

    for i in range(chunk_count):
        start_time = i * chunk_length_ms
        end_time = (i + 1) * chunk_length_ms
        chunk = source_audio[start_time:end_time]
        chunks.append(chunk)

    return chunks

def reduce_audio_size(before_modify_filename, before_export_filepath):
    fpath = os.path.join(UPLOAD_FOLDER, before_modify_filename)
    source_audio = AudioSegment.from_file(fpath)

    # 分割とリサンプリング
    chunk_length_ms = 60000  # 60秒毎に分割
    audio_chunks = split_and_reduce_audio(source_audio, chunk_length_ms)

    # 各チャンクを処理
    output_chunks = []
    for i, chunk in enumerate(audio_chunks):
        chunk_filename = f"chunk_{i + 1}.wav"
        chunk.export(os.path.join(UPLOAD_FOLDER, chunk_filename), format="wav")
        output_chunks.append(chunk_filename)

    return output_chunks


@whisper_module.route("/convert", methods=['GET', 'POST'])
def convert_audio_text():
    if request.method == 'POST':

        if 'file' not in request.files:
            messege = 'No file part.'
            return jsonify(messege), 400
        recieved_file = request.files['file']
        recieved_filename = recieved_file.filename

        if recieved_filename == '':
            messege = 'No selected file.'
            return jsonify(messege), 400
        
        if recieved_file and allowed_file(recieved_filename):
            hashing = lambda fname: str(uuid.uuid5(uuid.NAMESPACE_DNS, fname))

            filename, extention = recieved_filename.split(".")
            export_filename = secure_filename(hashing(filename) + F".{extention}")
            
            try:
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                export_filepath = os.path.join(UPLOAD_FOLDER, export_filename)
                recieved_file.save(export_filepath)
            except:
                messege = 'An error occurred while the file was being saved or transferred.'
                return jsonify(messege), 400
            
            audio_size = os.path.getsize(export_filepath)
            if audio_size > MAX_AUDIO_SIZE:
                # 音声データを分割してリサンプリング
                output_chunk_filenames = reduce_audio_size(export_filename, export_filepath)

                if len(output_chunk_filenames) == 0:
                    messege = 'Failed to process audio.'
                    os.remove(export_filepath)
                    return jsonify(messege), 400

                # 分割されたチャンクを個別にWhisperに送信して処理
                messages = []
                for chunk_filename in output_chunk_filenames:
                    try:
                        chunk_messege = callWhisper(chunk_filename)
                        messages.append(chunk_messege)
                    except:
                        messages.append('（一部音声の文字起こしに失敗しました）')

                # 分割されたファイルを削除
                for chunk_filename in output_chunk_filenames:
                    os.remove(os.path.join(UPLOAD_FOLDER, chunk_filename))

                combined_message = ''.join(messages)
                messege = [combined_message]
                os.remove(export_filepath)
                return jsonify(messege),200
            else:
                try:
                    messege = callWhisper(export_filename)
                except:
                    messege = 'An error occurred during speech recognition'
                    os.remove(export_filepath)
                    return jsonify(messege), 400
                    
                os.remove(export_filepath)
                return jsonify(messege), 200


    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

# 参考
# https://qiita.com/fghyuhi/items/d42ce8cb1f5de5280ac5
# https://msiz07-flask-docs-ja.readthedocs.io/ja/latest/patterns/fileuploads.html
