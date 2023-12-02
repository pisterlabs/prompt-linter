import Levenshtein
import openai
import dateparser
import spacy

def compute_spacy(text, nlp):
    towns = ['Douala', 'Yaoundé', 'Maroua', 'Garoua', 'Bafoussam', 'Bertoua', 'Bamenda', 'Ngaoundéré', 'Nkongsamba','Ebolowa',
             'Edea', 'Foumban', 'Buéa', 'Dschang', 'Limbé', 'Mbouda', 'Sangmélima', 'Bafang', 'Kousséri', 'Kribi']
    def similar_to_town(word,towns):
        for town in towns:
            similarity = 1- Levenshtein.distance(word,town)/max(len(word),len(town))
            if similarity >=0.42:
                return town
        return "Unknown"

    def get_destination(doc):
        for token in doc:
            if token.text == "to":
                # Get the next token in the doc
                next_token = doc[token.i + 1]
                # Check if the next token is an entity
                if next_token.ent_type == 384 or next_token.text in towns:
                    # Save the entity text in the destination variable
                    return next_token.text
                else:
                    return similar_to_town(next_token.text,towns)
        return "Unknown"

    def get_source(doc, destination):
        for ent in doc.ents:
            if (ent.label_ == "GPE" or ent.text in towns) and ent.text != destination:
                return ent.text
            else:
                return similar_to_town(ent.text,towns)
        return "Unknown"

    def get_schedule(doc):
        for ent in doc.ents:
            if ent.label_ == "DATE":
                return dateparser.parse(ent.text)
        return None

    doc = nlp(text)
    source = ""
    destination = ""
    schedule = ""
    
    destination = get_destination(doc)
    source = get_source(doc, destination)
    schedule = get_schedule(doc)

    return source, destination, schedule

def compute_openai(text, language = "en"):
    source = ""
    destination = ""
    schedule = ""

    if language == "fr":
        query = "The text I will provide is in french and I want you to extract the Source, Destination and Schedule from this text"
        queue = "The output should only be those three informations separated by commas"
    else:
        query = "I want you to extract the Source, Destination and Schedule from this text"
        queue = "The output should only be those three informations separated by commas"

    API_KEY = open('utilities/api_key.txt', 'r').read()
    openai.api_key = API_KEY
    response =openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{query}: '{text}' \n{queue}"}
        ]
    )

    print(response)
    return (source, destination, schedule)
