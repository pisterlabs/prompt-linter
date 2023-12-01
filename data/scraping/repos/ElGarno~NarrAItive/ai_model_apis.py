import requests
import base64
import openai
from elevenlabs import clone
from elevenlabs import set_api_key
from aws_scripts import get_voice_id_by_voice_name

CHUNK_SIZE = 1024


def get_completion_from_messages(api_key,
                                 messages,
                                 model="gpt-3.5-turbo-16k",
                                 temperature=0,
                                 max_tokens=3500):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def get_audio_from_text_and_save_to_file(api_key,
                                         text,
                                         speech_file_path,
                                         model="tts-1",
                                         voice="alloy"):
    client = openai.OpenAI(api_key=api_key)
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    response.stream_to_file(speech_file_path)


def get_image_from_prompt(api_key,
                          prompt,
                          model="dall-e-3"):
    client = openai.OpenAI(api_key=api_key)
    response = client.images.generate(
        prompt=prompt,
        model=model,  # Use the DALL-E model
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url
    return image_url


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_description_from_image(api_key,
                               image_url,
                               model="gpt-4-vision-preview"):
    # Path to your image
    image_path = image_url

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe the character in this image in about 2 sentences and don't talk about"
                                " the image itself. Just describe the character and start with "
                                "'A toddler, A boy.. / a girl / a man...' Make sure that you describe the colour of "
                                "hair and the potential age of the character."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = None
    for i_try in range(10):
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            break
    return response.json()["choices"][0]['message']['content']


def clone_voice(api_key, files, name, description="", labels=None):
    """
    Clone a voice from files.
    """
    if labels is None:
        labels = {}
    set_api_key(api_key)
    voice = clone(
        name=name,
        files=files,
        description=description,
        labels=labels
    )
    return voice


def get_voice_id_by_name(voices_response, name):
    return [voice["voice_id"] for voice in voices_response["voices"] if voice["name"] == name][0]


def generate_audio_voice_id(api_key, voice_id, text, file_path, model_id="eleven_multilingual_v2",
                            stability=0.5, similarity_boost=0.5):
    """
    Generate audio from a voice and a text.
    """

    # get voice_id by voice name
    # voice_id = get_voice_id_by_voice_name(voice_name)
    # voice_id = voice_name
    # print(f"voice_id: {voice_id}, file_path: {file_path}, text: {text}")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    response = requests.post(url, json=data, headers=headers)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def get_voices(api_key):
    set_api_key(api_key)
    url = "https://api.elevenlabs.io/v1/voices"
    response = requests.request("GET", url)
    return response.json()
