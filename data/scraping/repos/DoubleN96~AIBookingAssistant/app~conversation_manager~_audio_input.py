#import gradio as gr
#import openai, config, subprocess
#openai.api_key = os.environ.get('OPEN_API_KEY')
#
#messages = [
#                {"role": "system", "content": "You are a friendly room booking assistant."},
#            ]
#
#
#def transcribe(audio):
#    global messages
#
#    audio_file= open(audio, "rb")
#    transcript = openai.Audio.transcribe("whisper-1", audio_file)
#    print(transcript)
#
#    messages.append({"role": "user", "content": transcript["text"]})
#
#    response = openai.ChatCompletion.create(
#        model="gpt-3.5-turbo",
#        messages=messages
#        )
#    
#    system_message = choices[0]["message"]["content"]
#    messages.append(system_message)
#
#
#    #subprocess.call(["say", system_message['content']])
#
#    chat_transcript = ""
#    for message in messages:
#        if message['role'] != 'system':
#            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"
#
#    return chat_transcript
#
### demo = gr.Interface(fn=greet, inputs="text", outputs="text")
#ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text").launch()
#
#ui.launch()