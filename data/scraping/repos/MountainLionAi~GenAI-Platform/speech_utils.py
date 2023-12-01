from openai import OpenAI
from openai._types import FileTypes


def transcribe(
    client: OpenAI,
    file: FileTypes,
    prompt="ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T.",
) -> str:
    """
    把音频文件转成文本
    transcrible example:
    client = openai.OpenAI()
    with open(file_path,"rb") as audio_file
        text=transcribe(client,audio_file)
    """
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=file,
        prompt=prompt,
    )
    print(f"Transcribe:{transcript.text}")
    return transcript.text


def textToSpeech(client: OpenAI, text: str, file_name: str = "ai.mp3"):
    """
    把文本转成一个音频文件
    textToSpeech example:
    client = openai.OpenAI()
    textToSpeech(client,file_name="ai_temp.mp3")
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    response.stream_to_file(file=file_name)
