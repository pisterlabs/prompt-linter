from flask import Flask, render_template, request, redirect, url_for
import speech_recognition as sr
import openai
import pyttsx3

# Инициализация Flask
app = Flask(__name__)

# Инициализация библиотек и API-ключей
openai.api_key = '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'

engine = pyttsx3.init()

# Флаг для отслеживания состояния бота (слушает или нет)
listening = False

# Функция для озвучивания текста
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Функция для обработки голосовых команд
def listen_and_respond():
    global listening  # Используем глобальную переменную listening
    sr.pause_threshold = 0.8
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    
    try:
        query = r.recognize_google(audio, language='ru-RU')
        if "открой браузер" in query.lower():
            # Здесь вы можете выполнить действия, связанные с открытием браузера
            pass
        elif "открой ютуб" in query.lower() or "открой youtube" in query.lower():
            # Здесь вы можете выполнить действия, связанные с открытием YouTube
            pass
        # Добавьте другие условия для открытия разных сайтов
        else:
            completion = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role': 'user', 'content': query + "отвечай коротко"}
                ],
                temperature=0
            )
            response = completion['choices'][0]['message']['content']

            # Озвучиваем текст
            speak(response)
            return response
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        pass

@app.route('/')
def index():
    global listening
    return render_template('index.html', listening=listening)

@app.route('/start-listening', methods=['POST'])
def start_listening():
    global listening
    listening = True
    response = listen_and_respond()
    listening = False
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
