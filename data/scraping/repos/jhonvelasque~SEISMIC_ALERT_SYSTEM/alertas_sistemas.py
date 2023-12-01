import telebot
import config 
import openai
import pandas as pd
import time
import threading
#impementando el asistente 
openai.api_key = config.api_key
# Crear una instancia del bot con tu token
TOKEN = config.token
bot = telebot.TeleBot(TOKEN)
messages=[{"role":"system",
           "content":"eres un asistente que sabe mucho sobre acciones preventivas de sismo"}]

df=pd.read_csv('./data/muestra.csv')
df.sort_values(by=['time'], inplace=True, ascending=False)

def tipodesiniestro(num):
    if num in [1]:
        return '🚨🚨🚨SISMO EXTREMADAMENTE PELIGROSO🚨🚨🚨 '
    elif num in [2]:
        return '🚨🚨🚨PELIGROSO🚨🚨🚨'
    elif num in [3]:
        return '🚨🚨🚨MODERADO🚨🚨🚨'
    
contenido=f"""{tipodesiniestro(df.iloc[0,8])}
    -------------------------------------------------------------------------------
    🔹 el ultimo Movimiento telurico registrado fue {df.iloc[0,0][:10]} 
    con cordenadas {df.iloc[0,2],df.iloc[0,3]} cercano a {df.iloc[0,4]} ,
    con una magnitud de 😯{df.iloc[0,5]}.
    """
@bot.message_handler(commands=['sismo','ayuda'])
def amd_start(message):
    bot.reply_to(message, contenido)
# Manejar los mensajes de texto

texto_html = "<b>🙌HOLA ,ESTE ES UN BOT SISTEMAS DE SISMOS</b>"+"\n"
texto_html += """<i>Este es nuestro bot tenblorcin 😁 que te puede dar consejos de como actuar ante un temblor ,tambien te permite saber donde ocurrieron los temblores y de cual fue su magnitud y la gravedad de esta para ello solo dedes escribir los siguientes comandos .</i>"""+"\n" #cursiva
texto_html += "🔹 /sismo"+"\n" #surayado
texto_html += "🔹 /ayuda"+"\n"
texto_html += "<a href='https://github.com/jhonvelasque/SEISMIC_ALERT_SYSTEM.git'>🔗Entera mas del proyecto aqui </a>"+"\n" #link  

@bot.message_handler(content_types=['text'])
def bot_mensajes_texto(message):
    if message.text.startswith("/"):
        bot.send_message(message.chat.id,
                         """# Tener informacion de los sismos 
                            - /sismo
                            """,parse_mode='markdown')
    else : 
        messages.append({"role":"user","content":message.text})
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=messages)
        rta=response.choices[0].message.content
        bot.send_message(message.chat.id,rta,parse_mode='markdown')

def recibir_mensaje():
    #buvle infinito que comprueba lso mensajes
    bot.infinity_polling()
if __name__ == '__main__':
    print('Iniciando bot...')
    # Iniciar el bot y esperar mensajes
    hilo_bot=threading.Thread(name="hilo_bot",target=recibir_mensaje)
    hilo_bot.start()
    print('Inicio del chat')
    #mensaje al iniciar el chat bot
    foto=open('./picture/sismo.png','rb')
    bot.send_photo(config.chat_id,foto,'🎊🎊🎊bienvenido al grupo de alertas Sismicas🎊🎊🎊')
    bot.send_message(config.chat_id,texto_html,parse_mode='html')
    #leyendo el df