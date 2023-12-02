import openai
import av

def preprocess(video_path):
    input_container = av.open(video_path)
    input_stream = input_container.streams.get(audio=0)[0]

    output_container = av.open('code/uploads/live_stream.mp3', 'w')
    output_stream = output_container.add_stream('mp3')

    for frame in input_container.decode(input_stream):
        frame.pts = None
        for packet in output_stream.encode(frame):
            output_container.mux(packet)

    for packet in output_stream.encode(None):
        output_container.mux(packet)

    output_container.close()


def detect():
    audio_file= open("code/uploads/live_stream.mp3", "rb")
    print("Transcribing audio...")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print("Transcript: ", transcript)


def save():
    pass

def handle_conversation(video_name):
    video_path = 'code/uploads/{filename}'.format(filename=video_name)

    preprocess(video_path)

    return 'OK'