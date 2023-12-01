import os
import openai
from dotenv import load_dotenv
import pyshorteners

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def write_tweet(news_header: str, news_content: str, news_link: str):
    prompt = f"""

    Escríbeme un Twit actuando como si fueses Amador Rivas (personaje de la famosa serie "La Que se Avecina") muy exagerado
    para que gane mucha visibilidad en base a la siguiente noticia (riéndote de la noticia con mal carácter).
    La respuesta tiene que tener como máximo un total de 200 caracteres.
    Además no te repitas, intenta crear twits nuevos cada vez.

    Título: "{news_header}"
    
    Contenido:

    {news_content}

    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    
    shortened_news_link = pyshorteners.Shortener().tinyurl.short(news_link)
    
    return [f'{choice["message"]["content"]} {shortened_news_link}' for choice in response["choices"]]
