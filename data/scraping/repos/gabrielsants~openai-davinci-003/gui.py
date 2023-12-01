import PySimpleGUI as sg
import openai as ai
import pyttsx3

def requestResponse(question, chat_log = None) -> str:
    if(chat_log == None):
        chat_log = start_chat_log
    prompt = f"{chat_log}Human: {question}\nAI:"
    response = completion.create(
        prompt = prompt, 
        model="text-davinci-003",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop = "\nHuman: ")
    return response.choices[0].text

def convertToVoice(answer):
    engine = pyttsx3.init()
    engine.setProperty('rate',150)
    #comment this line if you are using linux or if you want to use your system default voice
    engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')
    engine.say(answer)
    engine.runAndWait()


if __name__ == "__main__":
    ai.api_key = 'YOUR_API_KEY'
    
    layout = [
        [sg.Text("Ask your question", font=('Helvetica', 24, 'bold'))],
        [sg.InputText(key="question", size=(45,10), font=('Arial', 14, 'bold'))],
        [sg.Button("Ask", size=(30,1)), sg.Button("Cancel", size=(29,1))],
        [sg.Text("", key="output")],
        [sg.Text("Open AI Davince GUI © 2023 Gabriel Santos")],
    ]
    
    completion = ai.Completion()
    start_chat_log = ""
    
    window = sg.Window("Open AI Davinci", layout)
    while True:
        event, values = window.read()
        
        if event == sg.WINDOW_CLOSED or event == "Cancel":
            break
        if event == "Ask":
            question = values["question"]
            answer = requestResponse(question,start_chat_log)
            window["output"].update(f"Answer: {answer}")
            convertToVoice(answer)
            