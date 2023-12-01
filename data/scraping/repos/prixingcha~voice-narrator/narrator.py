import os
import openai
import base64
import json
import time
import simpleaudio as sa
import errno
from elevenlabs import generate, play, set_api_key, voices
from dotenv import load_dotenv
import os

load_dotenv()
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


openai.api_key = OPENAI_API_KEY

# print(OPENAI_API_KEY)



set_api_key(ELEVENLABS_API_KEY)

def encode_image(image_path):
    while True:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            if e.errno != errno.EACCES:
                # Not a "file in use" error, re-raise
                raise
            # File is being written to, wait a bit and retry
            time.sleep(0.1)


def play_audio(text):
    # print(os.environ.get("ELEVENLABS_VOICE_ID"))
    # audio = generate(text, voice=os.environ.get("ELEVENLABS_VOICE_ID"))
    
    audio = generate(text, voice= ELEVENLABS_VOICE_ID)

    # unique_id = base64.urlsafe_b64encode(os.urandom(30)).decode("utf-8").rstrip("=")
    # dir_path = os.path.join("narration", unique_id)
    # os.makedirs(dir_path, exist_ok=True)
    # file_path = os.path.join(dir_path, "audio.wav")

    # with open(file_path, "wb") as f:
    #     f.write(audio)

    # play(audio)


def generate_new_line(base64_image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]


def analyze_image(base64_image, script):
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        # You are Sir David Attenborough. Narrate the picture of the human as if it is a nature documentary.
        messages=[
            {
                "role": "system",
                "content": """
                You are Matt Damon the famous actor and famous documentary narrator, generate a narration of documentary aboutMt. Everest of Nepal.
                Make it snarky and funny. Don't repeat yourself. Make it short. If I do anything remotely interesting, make a big deal about it!
                """,
            },
        ]
        + script
        + generate_new_line(base64_image),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content
    return response_text


def main():
    script = []

    while True:
        # strValue ="""
        #  You are Matt Damon the famous actor and famous documentary narrator, generate a narration of documentary about mount everst Mt. Everest of Nepal.
        #         Make it snarky and funny. Don't repeat yourself. also make it compelling narrative and authentic dialogue in the.
        # """
        
        strValue ="""
        Now, let’s talk about the history of this mountain. It was first conquered by Sir Edmund Hillary and Tenzing Norgay in 1953. And since then, it has become a popular destination for thrill-seekers and Instagram influencers alike. But let’s be real, most of us will never climb this mountain. We’ll just watch documentaries about it and pretend we’re there.
    Speaking of documentaries, did you know that there are more documentaries about Mount Everest than there are people who have actually climbed it? That’s right, you can watch hours and hours of people freezing their butts off and risking their lives for a chance to stand on top of a mountain. And if you’re lucky, you might even get to see someone take a dump in a bucket. Now that’s what I call entertainment.
    But in all seriousness, climbing Mount Everest is no joke. It’s dangerous, it’s expensive, and it’s not for the faint of heart. But if you’re up for the challenge, it can be a life-changing experience. Just don’t forget to bring a warm jacket and a selfie stick.
    I hope you enjoyed this snarky and funny documentary about Mount Everest. And remember, if you ever decide to climb this mountain, don’t forget to take a selfie at the top. It’s the ultimate flex.
        """
        
        # exit()
        # play_audio('')
        # exit()
        
        
        # path to your image
        image_path = os.path.join(os.getcwd(), "./frames/frame.jpg")

        # getting the base64 encoding
        base64_image = encode_image(image_path)

        # analyze posture
        print("👀 Matt is watching...")
        analysis = analyze_image(base64_image, script=script)

        print("🎙️ Matt says:")
        print(analysis)

        play_audio(analysis)

        script = script + [{"role": "assistant", "content": analysis}]

        # wait for 5 seconds
        # time.sleep(5)


if __name__ == "__main__":
    main()
