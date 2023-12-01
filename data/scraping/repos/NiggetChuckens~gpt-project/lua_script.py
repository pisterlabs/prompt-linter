import openai

openai.api_key=''

def translate(text:str,lang:str):
    prompt =(
        "You are going to be a good translator "
        "I need this text precisely in {} trying to keep the same meaning "
        "Translate from [START] to [END]:\n[START]"
    )
    prompt=prompt.format(lang)
    prompt += text + "\n[END]"
        
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        max_tokens=3000,
        temperature=0.4
    )
    return response.choices[0].text.strip()

def transcript_audio(audio_file:str, form:str, lan:str):
    audio_file=open(p(audio_file), 'rb') #audio file path
    transcript=openai.Audio.transcribe(
        file = audio_file,
        model ="whisper-1", 
        response_format=str(form),  #select output file format (srt, text)
        languaje=str(lan)   #Define languaje of the audio
        )
    return transcript
