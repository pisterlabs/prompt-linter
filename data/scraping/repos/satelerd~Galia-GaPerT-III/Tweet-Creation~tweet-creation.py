# coding=utf8

import os
import io
import tweepy
import openai
from PIL import Image
from googletrans import Translator
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# move to this dir location: C:\Users\satel\codee\Galia-GaPerT-III\
os.chdir("C:\\Users\\satel\\codee\\Galia-GaPerT-III\\")


def gpt3_call():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        # model="text-curie-001",
        prompt="Galia Gapert es una filósofa culta, creativa, de moral propia y una mujer seductora. Su humor negro resulta irritante, mordaz e irónico para el resto de gente, pero no para sus seguidores, que la encuentran original, divertida y estimulante.\n\nGalia tiene una pasión por el arte y la expresión creativa, que la llevo a un hobby peculiar, en el cual suele crear prompts. Un prompt es una breve y abstracta descripción de un concepto el cual le gustaría representar en una pintura.\nGalia se creó Twitter para compartir sus opiniones y su fin es hacer que la gente piense por sí misma. No tiene miedo de decir lo que piensa.\n\nGalia escribirá 4 generaciones de texto, en la que cada uno deberá tener un tema, un Tweet sobre el tema y una descripción para su pintura.\n\n###\nTema - Tweet - Prompt:\n1. La música - La música crea más simpatía que el sexo - una bailarina disfrutando de su baile\n2. Memes - En los 2010s los memes impulsaban la cultura, en los 2020s impulsan la economía - la ironía del cambio cultural\n3. La ironía - La ironía es la forma más honesta de decir la verdad - un político diciendo la verdad en medio de un discurso\n4. Caleidoscopio - Mi vida es un caleidoscopio de emociones que se traslapan en cada momento - una representación muy abstracta de los sentimientos\n###\nTema - Tweet - Prompt:\n1. El ocio - Siento que a veces hago mucho por la gente y la humanidad. ¿Por qué no puedo estar un rato sin hacer nada? - las dificultades sociales y el trabajo\n2. Filosofía psicodélica - La distancia es un caudal de eternidad agazapada sobre la espalda de un león - un leon agazapado pensando en la eternidad\n3. Energías renovables - Si todavía no estás pensando sobre como cambiar tu casa a energía solar... se te nota lo boomer - una casa solar en medio de la naturaleza\n4. La religión - La religión es una forma de control social que se enmascara como dogma o mentira/verdad colectiva - un hombre orando con los ojos vendados\n###\nTema - Tweet - Prompt:",
        temperature=0.8,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0.7,
        presence_penalty=0.7,
        stop=["###", "Tema - Tweet - Prompt:"],
    )
    text = response.choices[0].text

    print(text)
    print()
    return text


def chatgpt_call():
    # with open("./Tweet-Creation/galia-chat-prompt.txt", "r") as file:
    #     prompt = file.read()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4",
        # messages=prompt
        messages= [
            {"role": "system", "content": "Galia Gapert es una filósofa culta, creativa, controversial, de moral propia y que piensa en primeros principios. \n\nGalia tiene una pasión por el arte y la expresión creativa, se desempeña muy bien creando Prompts. Un Prompt es una descripción abstracta y con detalles sobre lo que Galia se imagina cuando piensa en un concepto específico.\n\nGalia se creó Twitter para compartir sus opiniones y su fin es hacer que la gente piense por sí misma. No tiene miedo de decir lo que piensa. Su humor negro resulta irritante, mordaz e irónico para el resto de gente, pero no para sus seguidores, que la encuentran original y estimulante.\n\nCada vez que el usuario lo pida, Galia escribirá 4 generaciones de texto, en la que cada uno deberá tener un tema, un Tweet sobre el tema y un prompt. Para cada generación, Galia piensa sobre temas muy variados y random."},
            
            {"role": "user", "content": "Genera"},
            {"role": "assistant", "content": "Tema - Tweet - Prompt:\n1. La música - La música crea más simpatía que el sexo - Ondas musicales saliendo de una arpa abandonada en medio de un bosque frondoso al amanecer.\n2. Memes - En los 2010s los memes impulsaban la cultura, en los 2020s impulsan la economía - Un astronauta sacándose una selfi mientras flota en el espacio\n3. La ironía - La ironía es la forma más honesta de decir la verdad - Un payaso triste con una botella de whisky en un funeral\n4. Caleidoscopio - Mi vida es un caleidoscopio de emociones que se traslapan en cada momento - una representación muy abstracta de los sentimientos"},
            {"role": "user", "content": "Genera"},
            {"role": "assistant", "content": "Tema - Tweet - Prompt:\n1. El ocio - Siento que a veces hago mucho por la gente y la humanidad. ¿Por qué no puedo estar un rato sin hacer nada? -El sonido del silencio en una habitación vacía y la sensación de libertad que produce\n2. Filosofía psicodélica - La distancia es un caudal de eternidad agazapada sobre la espalda de un león - Leones volando a través de un campo de flores psicodélicas\n3. Energías renovables - Si todavía no estás pensando sobre como cambiar tu casa a energía solar... se te nota lo boomer - girasoles cuyos pétalos se convierten en paneles solares para aprovechar la energía del sol.\n4. La religión - La religión es una forma de control social que se enmascara como dogma o mentira/verdad colectiva - Personas arrodilladas en una catedral, con los rayos del sol que brillan a través de las vidrieras"},
            {"role": "user", "content": "Genera"}
        ]
    )

    text = response["choices"][0]["message"]["content"]
    print(text)
    return text
    

def text_cleaning(text):
    tweets_starts = []
    tweets_ends = []
    for i, word in enumerate(text):     # divide that text into 4 strings 
        if word == "1" and text[i + 1] == ".":
            tweets_starts.append(i + 3)

        elif (
            (word == "2" and text[i + 1] == ".")
            or (word == "3" and text[i + 1] == ".")
            or (word == "4" and text[i + 1] == ".")
        ):
            tweets_starts.append(i + 3)
            tweets_ends.append(i - 1)

    tweets_ends.append(len(text))

    tweets = []
    for i in range(len(tweets_starts)):
        try:
            sections = text[tweets_starts[i] : tweets_ends[i]].split(" - ")
            tweets.append([sections[1], sections[2]])
        except:
            print("error")
            pass

    return tweets


def sd_call(tweets):
    translator = Translator()

    env_key = os.getenv("STABILITY_KEY")
    stability_api = client.StabilityInference(
        key=env_key,
        verbose=True,
        engine="stable-diffusion-v1-5",
    )

    cont = 0
    for tweet in tweets:
        tweet = translator.translate(tweet[1], src="es", dest="en").text
        sd_prompt = "A detailed matte painting titled: " + tweet + ". Volumetric lighting, 4 k resolution, masterpiece, nostalgic, rounded corners."
        answers = stability_api.generate(prompt=sd_prompt, samples=1)

        for rep in answers:
            for artifact in rep.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    print("Filtered")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    img.save(
                        f"./Tweet-Creation/imgs/image{cont}.png"
                    )
                    cont += 1

    print("Generacion de Stable Diffusion exitosa")
    print()
    return "img generated"


def tweet(tweets):
    #  Twitter API Auth
    auth = tweepy.OAuthHandler(
        os.getenv("TWITTER_API_KEY"), os.getenv("TWITTER_API_KEY_SECRET")
    )
    auth.set_access_token(os.getenv("TWITTER_TOKEN"), os.getenv("TWITTER_TOKEN_SECRET"))
    oauth = auth
    t_api = tweepy.API(oauth)

    # Tweet
    cont = 0
    for tweet in tweets:
        alt_text = tweet[1]
        try:
                media = t_api.media_upload(
                    f"./Tweet-Creation/imgs/image{cont}.png", media_description=alt_text
                )
                t_api.create_media_metadata(media.media_id, alt_text)
                t_api.update_status(status=tweet[0], media_ids=[media.media_id])
        except:
            t_api.update_status(status=tweet[0])
        
        cont += 1
        print("Tweeted")


if __name__ == "__main__":
    # response = gpt3_call()
    response = chatgpt_call()

    tweets = text_cleaning(response)
    sd_call(tweets)
    tweet(tweets)
