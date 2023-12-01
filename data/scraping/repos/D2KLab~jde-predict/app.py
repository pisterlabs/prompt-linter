from flask import Flask
from flask_restx import Resource, Api, fields, reqparse
import requests
import anthropic
from dotenv import load_dotenv
import re
import os
import openai
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import redis
import json


load_dotenv()

redis_client = redis.Redis(host=os.getenv('REDIS_HOST'), decode_responses=True)

jde_classes = [
    "Rachat / Cession",
    "Levée de fonds",
    "Nouvelle implantation",
    "Changement de Dirigeant",
    "Procédure de sauvegarde",
    "Fermeture de site",
    "Création d’emploi / recrutement",
    "Extension géographique",
    "Investissement",
    "Nouvelle activité / produit",
    "Projet d’acquisition"
]
zeste_config = {
    'classes': ['rachat', 'bienfaisance', 'implantation', 'passation', 'banqueroute', 'fermeture', 'recrutement', 'territoire', 'investissement', 'innovation', 'acquisition'],
    'threshold': 0.11, # float or None
    'topk': 3, # int or None
}

app = Flask(__name__)
api = Api(app)

PredictRequest = reqparse.RequestParser()
PredictRequest.add_argument('method', choices=['bert', 'claude-v1', 'gpt-4', 'zeste'], required=True)
PredictRequest.add_argument('url', type=str, location='form')
PredictRequest.add_argument('text', type=str)
Prediction = api.model('Prediction', {
    'label': fields.String,
    'score': fields.Float,
})
PredictResponse = api.model('PredictResponse', {
    'method': fields.String,
    'predictions': fields.List(fields.Nested(Prediction)),
})

ThemesRequest = reqparse.RequestParser()
ThemesRequest.add_argument('url', type=str, location='form')
ThemesRequest.add_argument('text', type=str)
ThemesResponse = api.model('ThemesResponse', {
    'themes': fields.List(fields.String),
})

EntitiesRequest = reqparse.RequestParser()
EntitiesRequest.add_argument('url', type=str, location='form')
EntitiesRequest.add_argument('text', type=str)
EntitiesResponse = api.model('EntitiesResponse', {
    'html': fields.String,
    'entities': fields.Raw,
})

def get_text_from_url(url):
    cached_text = redis_client.get(f'texts|{url}')
    if cached_text is not None:
        print(f'[TEXT][CACHE] {url}')
        return cached_text

    print(f'[TEXT][QUERY] {url}')
    parsed_url = urlparse(url)
    if parsed_url.netloc != 'www.lejournaldesentreprises.com':
        raise ValueError('Invalid url')
    res = requests.get(url, params={'_format': 'json'})
    data = res.json()
    texts = []
    if 'body' in data and len(data['body']) > 0 and 'value' in data['body'][0]:
        html = data['body'][0]['value']
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        texts.append(text)
    if 'field_abstract' in data and len(data['field_abstract']) > 0 and 'value' in data['field_abstract'][0]:
        texts.append(data['field_abstract'][0]['value'])
    final_text = '. '.join(texts)

    redis_client.set(f'texts|{url}', final_text)
    return final_text


def get_zeste_predictions(text):
    assert(len(zeste_config['classes']) == len(jde_classes))
    res = requests.post('https://zeste.tools.eurecom.fr/api/predict',
        headers={
            'accept': 'application/json',
        },
        json={
            "labels": zeste_config['classes'],
            "language": "fr",
            "text": text,
            "explain": False,
            "highlights": False
        }
    )
    json = res.json()
    predictions = []
    for i, result in enumerate(json['results']):
        # Check if i if above topk
        if zeste_config['topk'] is not None and i >= zeste_config['topk']:
            break
        # Check if result['score'] is above threshold
        if zeste_config['threshold'] is not None and result['score'] < zeste_config['threshold']:
            continue
        # Get index of class from zeste_classes based on result['label']
        index = zeste_config['classes'].index(result['label'])
        # Get corresponding class from jde_classes
        jde_class = jde_classes[index]
        predictions.append({
            'label': jde_class,
            'score': result['score']
        })
    return predictions


def get_claude_predictions(text):
    claude_classes = [
        'Rachat / Cession',
        'Levée de fonds',
        'Nouvelle implantation',
        'Changement de Dirigeant',
        'Procédure de sauvegarde',
        'Fermeture de site',
        "Création d'emploi / recrutement",
        'Extension géographique',
        'Investissement',
        'Nouvelle activité / produit',
        "Projet d'acquisition"
    ]
    assert(len(jde_classes) == len(claude_classes))
    prompt_classes = '\n'.join([f'{i+1}. {x}' for i, x in enumerate(claude_classes)])

    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
    client = anthropic.Client(CLAUDE_API_KEY)

    text = text.replace('\n', ' ')[0:28000-500]
    prompt=f"""{anthropic.HUMAN_PROMPT}Texte à classifier: {text}.

Veuillez retourner jusqu'à 3 numéros de catégories séparés par une virgule, parmi les options suivantes, uniquement si explicitement décrites.

{prompt_classes}

Choix:{anthropic.AI_PROMPT}"""
    resp = client.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1.3",
        max_tokens_to_sample=8,
    )
    predictions = []
    numbers = re.findall(r'\d+', resp['completion'])
    numbers_list = [int(num) for num in numbers]
    for num in numbers_list:
        if num >= 1 and num <= len(jde_classes):
            predictions.append({
                'label': jde_classes[num - 1],
                'score': 1.0
            })
    return predictions


def get_gpt_predictions(text):
    gpt_classes = [
        "Rachat / Cession",
        "Levée de fonds",
        "Nouvelle implantation",
        "Changement de Dirigeant",
        "Procédure de sauvegarde",
        "Fermeture de site",
        "Création d'emploi / recrutement",
        "Extension géographique",
        "Investissement",
        "Nouvelle activité / produit",
        "Projet d'acquisition"
    ]
    assert(len(jde_classes) == len(gpt_classes))
    prompt_classes = '\n'.join([f'{i+1}. {x}' for i, x in enumerate(gpt_classes)])

    openai.api_key = os.getenv('OPENAI_API_KEY')

    text = text.replace('\n', ' ')[0:8000-500]
    prompt=f"""Texte à classifier: {text}.

Veuillez retourner jusqu'à 3 numéros de catégories séparés par une virgule, parmi les options suivantes, uniquement si explicitement décrites.

{prompt_classes}

Choix:"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            { "role": "user", "content": prompt },
        ]
    )

    predictions = []
    numbers = re.findall(r'\d+', response['choices'][0]['message']['content'])
    numbers_list = [int(num) for num in numbers]
    for num in numbers_list:
        if num >= 1 and num <= len(jde_classes):
            predictions.append({
                'label': jde_classes[num - 1],
                'score': 1.0
            })

    return predictions


def get_bert_predictions(text):
    res = requests.post(os.getenv('BERT_API_URL') + '/predict',
        headers={
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
        },
        data={
            "text": text,
        }
    )
    data = res.json()
    return data['predictions']


def get_claude_themes(text):
    claude_classes = [
        'Fusion - Acquisition',
        'RSE',
        'Ressource Humaine',
        'Emploi',
        # 'Fiscalité',
        # 'Juridique'
        'International',
        'Carnet',
        'Investissement',
        'Projet',
    ]
    prompt_classes = '\n'.join([f'{i+1}. {x}' for i, x in enumerate(claude_classes)])

    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
    client = anthropic.Client(CLAUDE_API_KEY)

    text = text.replace('\n', ' ')[0:28000-2000]
    prompt=f"""{anthropic.HUMAN_PROMPT}Texte à classifier: {text}.

Veuillez retourner le numéro de la catégorie, parmi les options suivantes. Si plusieurs catégories sont détectées, retourner les deux premières. Si aucune catégorie n'est reconnue, retourner NULL.

{prompt_classes}

Choix:{anthropic.AI_PROMPT}"""
    resp = client.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1.3",
        max_tokens_to_sample=8,
    )
    themes = []
    numbers = re.findall(r'\d+', resp['completion'])
    numbers_list = [int(num) for num in numbers]
    for num in numbers_list:
        if num >= 1 and num <= len(claude_classes):
            themes.append(claude_classes[num - 1])
    return themes


@api.route('/status')
class Status(Resource):
    @api.response(200, 'Success')
    def get(self):
        return 'OK'


@api.route('/predict')
class Predict(Resource):
    @api.expect(PredictRequest)
    @api.marshal_with(PredictResponse)
    @api.response(200, 'Success')
    @api.response(501, 'Prediction method not implemented')
    def post(self):
        args = PredictRequest.parse_args()
        method = args['method']
        url = args['url']
        text = args['text']

        if url is None and text is None:
            return 'Either url or text must be provided', 400

        if url is not None and text is not None:
            return 'Only one of url or text must be provided', 400

        if url is not None:
            text = get_text_from_url(url)

        # Get unique hash from text string
        text_hash = hash(text)

        # Check if predictions are cached
        cached_predictions = redis_client.get(f'predictions|{method}|{text_hash}')
        if cached_predictions is not None:
            print(f'[PREDICT][CACHE] {method} {text_hash}')
            return { 'predictions': json.loads(cached_predictions) }

        # Predict
        print(f'[PREDICT][QUERY] {method} {text_hash}')
        if method == 'zeste':
            predictions = get_zeste_predictions(text)
        elif method == 'claude-v1':
            predictions = get_claude_predictions(text)
        elif method == 'gpt-4':
            predictions = get_gpt_predictions(text)
        elif method == 'bert':
            predictions = get_bert_predictions(text)
        else:
            return 'Prediction method not implemented', 501

        redis_client.set(f'predictions|{method}|{text_hash}', json.dumps(predictions))
        return { 'method': method, 'predictions': predictions }


@api.route('/entities')
class Predict(Resource):
    @api.expect(EntitiesRequest)
    @api.marshal_with(EntitiesResponse)
    @api.response(200, 'Success')
    def post(self):
        args = EntitiesRequest.parse_args()
        url = args['url']
        text = args['text']

        if url is None and text is None:
            return 'Either url or text must be provided', 400

        if url is not None and text is not None:
            return 'Only one of url or text must be provided', 400

        if url is not None:
            text = get_text_from_url(url)

        text_hash = hash(text)

        cached_entities = redis_client.get(f'entities|{text_hash}')
        if cached_entities is not None:
            print(f'[ENTITIES][CACHE] {text_hash}')
            return json.loads(cached_entities)

        print(f'[ENTITIES][QUERY] {text_hash}')

        res = requests.post(os.getenv('BERT_API_URL') + '/ner',
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
            },
            data={
                "text": text,
            }
        )
        data = res.json()
        html = data['html']
        entities = data['entities']

        response = { 'html': html, 'entities': entities }
        redis_client.set(f'entities|{text_hash}', json.dumps(response))
        return response


@api.route('/themes')
class Themes(Resource):
    @api.expect(ThemesRequest)
    @api.marshal_with(ThemesResponse)
    @api.response(200, 'Success')
    def post(self):
        args = EntitiesRequest.parse_args()
        url = args['url']
        text = args['text']

        if url is None and text is None:
            return 'Either url or text must be provided', 400

        if url is not None and text is not None:
            return 'Only one of url or text must be provided', 400

        if url is not None:
            text = get_text_from_url(url)

        text_hash = hash(text)

        cached_themes = redis_client.get(f'themes|{text_hash}')
        if cached_themes is not None:
            print(f'[THEMES][CACHE] {text_hash}')
            return json.loads(cached_themes)

        print(f'[THEMES][QUERY] {text_hash}')

        themes = get_claude_themes(text)

        response = { 'themes': themes }
        redis_client.set(f'themes|{text_hash}', json.dumps(response))
        return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('PORT', '5000'))