import os
import sys
import argparse
import asyncio
import typer
from datetime import datetime
from ConnectionDao.mongodb_connection import MongoDBConnection
from ClassApi.news import News
from ClassApi.twitter import Twitter
from ClassApi.openai import OpenAI
from ClassApi.telegram import TelegramBot
from ClassApi.google import GoogleSearcher
from ClassLogManager.log_manager import LogManager

# Configuración
setting = MongoDBConnection.find_documents('setting')
for configs in setting:
    id_setting = configs['_id']
    banned_words = configs['bannedwords']
    spacy_model_default = configs['spacy_model']['large']
    access_key = configs['token_user_twitter']['access_key']
    access_secret = configs['token_user_twitter']['access_secret']
    api_key_news = configs['apikeynews']
    api_key_cutt = configs['apikeycutt']
    api_key_openai = configs['apikeyopenai']
    api_key_telegram = configs['apikeytelegram']['bot_token']
    api_key_google = configs['apikeygooglesearch']['api_key']
    api_cx_google = configs['apikeygooglesearch']['cx']

# Crea una instancia del google
google_instance = GoogleSearcher(api_key_google, api_cx_google)
# Creación del objeto News
news = News(api_key_news, api_key_cutt, banned_words)

# Creación del objeto Twitter
twitter = Twitter(id_setting, access_key, access_secret)

# Crear instancia de la clase OpenAI y configurar la API key
openai_instance = OpenAI(api_key_openai)

# Crea una instancia del bot de Telegram
telegram_bot = TelegramBot(api_key_telegram, openai_instance, spacy_model_default)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--authorization",
        action = 'store_true',
        required = False,
        help = "Obtiene los token autorización del usuario."
    )
    parser.add_argument(
        "-news",
        "--postnew",
        type = str,
        help = "Publica noticias en twiter."
    )
    parser.add_argument(
        "-l",
        "--language",
        choices = ['en', 'es'],
        default = 'es',
        type = str,
        required = False,
        help = "Especifica un idioma para la busqueda 'en' (English) o 'es' (Español)"
    )
    parser.add_argument(
        "-fs",
        "--followers",
        action = 'store_true',
        required = False,
        help = "Lista de seguidores."
    )
    parser.add_argument(
        "-fw",
        "--following",
        action = 'store_true',
        required = False,
        help = "Lista de seguidos."
    )
    parser.add_argument(
        "-ff",
        "--followback",
        action = 'store_true',
        required = False,
        help = "Lista de seguidos de vuelta (Followback)."
    )
    parser.add_argument(
        "-fb",
        "--nofollowback",
        action = 'store_true',
        required = False,
        help = "Lista de no seguidos de vuelta (Notfollowback)."
    )
    parser.add_argument(
        "-bck",
        "--blocked",
        action = 'store_true',
        required = False,
        help = "Lista de cuentas bloqueadas."
    )
    parser.add_argument(
        "-m",
        "--muted",
        action = 'store_true',
        required = False,
        help = "Lista de cuentas silenciadas."
    )
    parser.add_argument(
        "-o",
        "--chatgpt",
        action = 'store_true',
        required = False,
        help = "Activa ChatGpt."
    )
    parser.add_argument(
        "-t",
        "--telegrambot",
        action = 'store_true',
        required = False,
        help = "Activa TelegramBot."
    )

    return parser.parse_args()

# Función principal
async def main():
    args = parse_args()

    action_mapping = {
        'authorization': twitter.get_authorization,
        'postnew': lambda: twitter.set_tweet(news.search_news(args.postnew, args.language)),
        'followers': lambda: twitter.get_user_data('followers', twitter.get_followers()),
        'following': lambda: twitter.get_user_data('following', twitter.get_following()),
        'nofollowback': lambda: twitter.get_user_data('nofollowback', twitter.nofollowback()),
        'followback': lambda: twitter.get_user_data('followback', twitter.followback()),
        'blocked': lambda: twitter.get_user_data('blockedaccounts', twitter.get_blockedaccounts()),
        'muted': lambda: twitter.get_user_data('mutedaccounts', twitter.get_mutedsaccounts()),
        'chatgpt': lambda: openai_instance.get_response(input('Ingrese su pregunta: ')),
        'telegrambot': telegram_bot.run
    }

    executed = False
    for arg, func in action_mapping.items():
        if getattr(args, arg):
            if arg == "telegrambot":
                await func()
            else:
                result = func()
                print(f"Resultado de {arg}: {result}")
            executed = True

    if not executed:
        print(f'Seleccione una opción: -h para ayuda')

if __name__ == "__main__":
    print('*' * 30, 'INICIO EJECUCION BOTS', '*' * 30)
    LogManager.log("INFO", "INICIO EJECUCION BOTS")

    try:
        asyncio.run(main())
    except Exception as e:
        LogManager.log("ERROR", f'Error de Ejecucion: {e}')
    finally:
        LogManager.log("INFO", f'FIN - EJECUCION BOTS')
        print('*' * 30, 'FIN - EJECUCION BOTS', '*' * 30)
        sys.exit()
