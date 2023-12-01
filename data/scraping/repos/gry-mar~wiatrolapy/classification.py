import openai


api_key = 'api_key'

openai.api_key = api_key

topics = [
    "aborcja", "tabletka dzień po",
    "500plus", "800plus",
    "prawa kobiet", "równość płac", "feminizm",
    "lgbt",
    "Imigranci", "relokacja",
    "granica polsko-białoruska", "płot",
    "Rosja",
    "Ukraina",
    "USA", "Stany Zjednoczone",
    "Unia Europejska",
    "Inflacja",
    "TVP", "telewizja polska", "telewizja publiczna",
    "TVN",
    "Kościół",
    "NFZ", "publiczna opieka zdrowotna",
    "Covid",
    "Edukacja seksualna",
    "Podatki",
    "Emerytura",
    "In-vitro",
    "ZUS",
    "sądownictwo",
    "węgiel",
    "marihuana"
]


def analyze_text(text: str) -> str:

    content = """Classify the provided text into one of the provided classes and determine the sentiment towards that class. If the text covers multiple topics, return appropriately classified fragments of provided text for each topic along with sentiment analysis. If none of the listed topics apply, use your judgment to determine the topic based on context. Return the results as a list of JSON objects, where each object has fields: 
    {"text" : provided text or text fragment,  
    "topic" : determined topic,
    "sentiment": sentiment}""" + f" Classes: {topics} Text: {text}"
    
    if len(text) <= 10000:
        model = 'gpt-3.5-turbo'
    elif len(text) > 10000 and len(text) < 46000:
        model = 'gpt-3.5-turbo-16k'
    else:
        model = "gpt-4-1106-preview"

    response = openai.chat.completions.create(
        model=model,
        temperature=0.6,
        messages=[
            {"role": "user", "content": content},
        ]
        )
    return response.choices[0].message.content



