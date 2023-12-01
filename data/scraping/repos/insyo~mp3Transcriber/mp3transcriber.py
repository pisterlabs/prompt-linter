import json
import os
import sys
import wave

import openai
from pydub import AudioSegment
from vosk import KaldiRecognizer, Model


def transcribe_audio(mp3_path, model_path) -> str:
    """
    mp3ファイルを指定されたVOSK Modelで音声認識し、認識結果のテキストを返す
    """
    # mp3をwavに変換
    audio = AudioSegment.from_mp3(mp3_path)
    wav_path = "temp.wav"
    if audio.channels > 1:
        audio = audio.set_channels(1)  # mono PCMでないと音声認識ができない(重要!!!)
    audio.export(wav_path, format="wav", codec="pcm_s16le")

    # 念のため音声形式(mono PCM)のチェック
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise Exception("Audio file must be WAV format mono PCM.")

    # Model(model_path)で呼び出されるC言語実装ライブラリのstdout/stderrを抑止するために、/dev/nullにリダイレクトする
    orig_stdout_fd = os.dup(1)
    orig_stderr_fd = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)

    # VOSK Modelを読み込む
    model = Model(model_path)

    # stdout/stderrを元に戻す
    os.dup2(orig_stdout_fd, 1)
    os.dup2(orig_stderr_fd, 2)
    os.close(orig_stdout_fd)
    os.close(orig_stderr_fd)

    # 音声認識エンジンを初期化
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    rec.SetPartialWords(True)

    text = ""

    # 音声ファイルを読み込みながら音声認識を実行
    print("Transcribing", end="", flush=True)
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:  # 音声ファイルからデータが読めなくなったら終わりにする
            break
        if rec.AcceptWaveform(data):  # 取り出した音声データを文字起こし
            text += json.loads(rec.Result())["text"]
            print(".", end="", flush=True)

    print("Done!")
    os.remove(wav_path)

    return text


def format_text_by_llm(text: str):
    """
    ChatGPTにて、文章を読みやすい形式に整形する
    """

    PROMPT_TEMPLATE = """
    下記の文章を読みやすい形式に整形してください。
    語尾、接続詞、句読点を修正するだけで、要約や記述の内容は変更しないでください。

    ---
    {text}
    """

    # プロンプトの組み立て
    prompt = PROMPT_TEMPLATE.format(text=text)

    # LLM Modelの生成(stream=True指定の場合はgeneratorが返す)
    generator = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "あなたは優秀で忠実な翻訳者です。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        stream=True,
    )

    # generatorからのレスポンス(=整形済文章)を随時受け取り、随時返却(yield)する
    for chunk in generator:
        response = chunk["choices"][0]["delta"]  # type: ignore
        if "content" in response:
            yield response["content"]

    return None


def main():
    # 引数チェック
    if len(sys.argv) != 2:
        print(f"python3 {sys.argv[0]} <mp3_file>")
        return
    mp3_file = sys.argv[1]

    # 環境変数チェック
    if "OPENAI_API_KEY" not in os.environ:
        print('Environment variable "OPENAI_API_KEY" is not set.')
        return
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # 音声認識
    model_path = "./vosk-model-ja-0.22"
    transcribed_text = transcribe_audio(mp3_file, model_path)

    # 文章整形
    for chunk_text in format_text_by_llm(transcribed_text):
        print(chunk_text.replace("。", "。\n"), end="", flush=True)  # 「。」で改行しながら出力
    print()


if __name__ == "__main__":
    main()
