import openai
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from os import remove


# Token válido por 24h acesse esse site https://platform.openai.com/account/api-keys
openai.api_key = "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"


def input_voz():
    """
    Função para reconhecimento de voz do usuário.
    """
    rec = sr.Recognizer()
    with sr.Microphone() as mic:
        rec.adjust_for_ambient_noise(mic)
        text_voz("Pergunte-me algo.")
        audio = rec.listen(mic)
    try:
        texto = rec.recognize_google(audio, language="pt-BR")
        print("Você disse:", texto)
        return texto
    except sr.UnknownValueError:
        text_voz("Não entendi. Tente novamente.")


def text_voz(txt):
    """
    Função para transformar texto em voz.
    """
    try:
        tts = gTTS(text=txt, lang='pt', tld='com.br', slow=False)
        tts.save('texto.mp3')
        playsound('texto.mp3')
        remove('texto.mp3')
    except Exception as e:
        print(f"Alguma coisa deu errado no text_voz: {e}")


def ask_question(prompt):
    """
    Função para enviar a pergunta do usuário para a API da OpenAI e obter a resposta.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        n=1,
        temperature=0.5,
    )

    answer = response.choices[0].text.strip()
    return answer


while True:
    user_input = input_voz()
    response = ask_question(user_input)
    text_voz(response)
    
