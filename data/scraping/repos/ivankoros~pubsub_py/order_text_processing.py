import re
import nltk.corpus
from nltk.corpus import stopwords, wordnet
from backend.twilio_app.helpers import all_sandwiches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import ast
from dotenv import load_dotenv


def clean_text(user_text_input):
    #nltk.download('stopwords')
    text = user_text_input.lower()

    # Everything that isn't a letter or space is removed
    text = re.sub(r"[^a-zA-Z\s']", "", text)

    # Remove stop words (e.g. "the", "a", "an", "in")
    stop_words = stopwords.words('english')
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text


def find_synonyms(word):
    #nltk.download('wordnet')
    """Find synonyms of a word using WordNet

    This function is for identifying the meaning of the user's input,
    although they may not have used the correct words. For example,
    if the user says a "meatball hoagie", we still want to be able to
    find and return the "Publix Meatball Sub" sandwich.

    This function takes in a word and follows this process:
    1. Loop over each synonym (syn) in the synset (the group of synonyms) of the word.
    2. Loop over each lemma (the base form of a word) in the synset.
    3. For each lemma, add it to the set of synonyms.

    :param word: string, single word
    :return: list of synonyms for the given word
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def vectorized_string_match(item_to_match, match_possibilities_list=all_sandwiches):

    if not item_to_match:
        return None

    user_input_clean = clean_text(item_to_match)

    """Convert the sandwich names to TF-IDF vectors

    What I'm doing here is converting each sandwich name into a vector of numbers.
    Each number represents how important that word is to the sandwich name.

    The most highly rated words should be the ones found in our list of subs, so what I do
    is pass the list of subs into the model which will calculate the scores for each word,
    placing more emphasis on the words that are found in the subs.

    This is important because the user's input should be natural:
        "I'm feeling like getting a hot italian today"

    Here, the word "italian", which is in our list of subs, needs to be rated extremely
    high.

    """

    # Initialize the vectorizer and fit it to the list of subs for our model
    vectorizer = TfidfVectorizer()
    sandwich_vectors = vectorizer.fit_transform(match_possibilities_list)

    # Vectorize the user's input and compare it against the vectorized subs list
    user_vector = vectorizer.transform([user_input_clean])
    similarity_scores = cosine_similarity(user_vector, sandwich_vectors)[0]

    # Find the highest score and return the corresponding sub by index
    best_match_index = similarity_scores.argmax()
    best_match_score = similarity_scores[best_match_index]
    best_match = match_possibilities_list[best_match_index]

    return best_match if best_match_score >= 0.5 else None


def parse_customizations(user_order_text):
    env_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", 'config\\.env'))

    load_dotenv(dotenv_path=env_directory,
                override=True)

    openai.api_key = os.getenv("OPEN_AI_API_KEY")

    messages = [
        {
            "role": "system",
            "content": (
                "Based on the following list of customization options: \n"
                "Size: Half, Whole \n"
                "Bread: Italian 5 Grain, White, Whole Wheat, Flatbread, No Bread (Make it a Salad) - Lettuce Base, No Bread (Make it a Salad) - Spinach Base\n"
                "Cheese: Pepper Jack, Cheddar, Muenster, Provolone, Swiss, White American, Yellow American, No Cheese\n"
                "Extras: Double Meat, Double Cheese, Bacon, Guacamole, Hummus, Avocado\n"
                "Toppings: Banana Peppers, Black Olives, Boar's Head Garlic Pickles, Cucumbers, Dill Pickles, Green Peppers, Jalapeno Peppers, Lettuce, Onions, Spinach, Tomato, Salt, Black Pepper, Oregano, Oil & Vinegar Packets\n"
                "Condiments: Boar's Head Honey Mustard, Boar's Head Spicy Mustard, Mayonnaise, Yellow Mustard, Vegan Ranch Dressing, Buttermilk Ranch, Boar's Head Sub Dressing, Boar's Head Pepperhouse Gourmaise, Boar's Head Chipotle Gourmaise, Deli Sub Sauce\n"
                "Heating Options: Pressed, Toasted, No Thanks\n"
                "Make it a Combo: Yes, No Thanks\n"
                "\n"
                "If a field is not specified, reply with 'None' Only. There should only be 8 category outputs, absolutely no more. If there are duplicates (for example, "
                " two 'condiments' rows), consolidate them.\n"
                "If a user has 'packets' in their message, they want the 'Oil & Vinegar Packets' topping\n"
                "Follow this rule strictly: Return customization options exactly as they appear in the text. For example 'sub oil' should return 'Deli Sub Sauce'\n"
                "\n"
                "Give back a list of selected customization options from the user's text input, which is a sandwich order\n\n"
                "Return the list back as a Python dictionary, with the keys being the category and the values being the selected options. \n"
            ),
        },
        {"role": "user", "content": f"Sandwich order: {user_order_text}"}
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.1,
    )

    response = completion.choices[0].message['content']

    try:
        return ast.literal_eval(response)
    except (ValueError, SyntaxError) as e:
        print("Error:", e)


def find_sub_match(user_order_text):

    env_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", 'config\\.env'))

    load_dotenv(dotenv_path=env_directory,
                override=True)

    openai.api_key = os.getenv("OPEN_AI_API_KEY")

    messages = [
        {
            "role": "system",
            "content": (
                "You'll be given a user's sandwich order. Determine which of the sandwiches from the following "
                "sandwich_list is the closest match to the user's order. \n"
                ""
                "Follow this rule: If no match is found, return None. \n"
                "Follow this rule: Return the exact name of the most likely match exactly as it appears in the "
                "list as a string. For example, if the user says 'I want a Boar's Head Turkey sub with chipotle sauce, onions, tomatos, mayonais, olives, and cucumbers on italian grain bread', only return: 'Boar's Head Turkey Sub' \n"

                "Here is the list of sandwiches the user can choose from: ["
                "Boar's Head Ultimate Sub, "
                "Boar's Head Turkey Sub, "
                "Publix Italian Sub, "
                "Publix Chicken Tender Sub, "
                "Boar's Head Italian Sub, "
                "Publix Turkey Sub, "
                "Boar's Head Ham Sub, "
                "Boar's Head Italian Wrap, "
                "Publix Veggie Sub, "
                "Boar's Head Maple Honey Turkey Sub, "
                "Boar's Head Roast Beef Sub, "
                "Publix Ultimate Sub, "
                "Publix Turkey Wrap, "
                "Boar's Head Ultimate Wrap, "
                "Publix Veggie Wrap, "
                "Chicken Cordon Bleu Sub Hot, "
                "Boar's Head Honey Turkey Wrap, "
                "Boar's Head Jerk Turkey & Gouda Sub, "
                "Publix Tuna Salad Sub, "
                "Publix Homestyle Beef Meatball Sub, "
                "Publix Italian Wrap, "
                "Boar's Head Roast Beef Wrap, "
                "Publix Ham Sub, "
                "Boar's Head Ham and Turkey Sub, "
                "Publix Deli Spicy Falafel Sub Wrap Hot, "
                "Publix Roast Beef Sub, "
                "Publix Ultimate Wrap, "
                "Publix Chicken Salad Wrap, "
                "Boar's Head Everroast Wrap, "
                "Publix Deli Spicy Falafel Sub, "
                "Boar's Head Philly Cheese Sub, "
                "Boar's Head Havana Bold Sub, "
                "Boar's Head EverRoast Sub, "
                "Publix Deli Tex Mex Black Bean Burger Sub, "
                "Publix Ham Wrap, "
                "Boar's Head Low Sodium Ultimate Sub,"
                "Boar's Head Philly Wrap, "
                "Publix Ham & Turkey Sub, "
                "Publix Greek Sub, "
                "Publix Deli Meatless Turkey Club Sub, "
                "Boar's Head Ham Wrap, "
                "Publix Roast Beef Wrap, "
                "Publix Egg Salad Wrap, "
                "Publix Deli Baked Chicken Tender Wrap, "
                "Publix Deli Baked Chicken Tender Sub, "
                "Publix Tuna Salad Wrap, "
                "Boar's Head BLT Hot Sub, "
                "Boar's Head Low Sodium Turkey Sub, "
                "Boar's Head Blt Wrap, "
                "Publix Chicken Salad Sub, "
                "Publix Philly Cheese Sub, "
                "Publix Egg Salad Sub, "
                "Publix Deli Meatless Turkey Club Sub Wrap, "
                "Publix Deli Tex Mex Black Bean Burger Wrap, "
                "Publix Cuban Sub, "
                "Publix Deli Ham Salad Sub, "
                "Boar's Head Cajun Turkey Sub, "
                "Boar's Head Chipotle Chicken Wrap, "
                "Boar's Head Cracked Pepper Turkey Sub, "
                "Publix Deli Nashville Hot Chicken Tender Su, "
                "Publix Deli Nashville Hot Chicken Tender Wr, "
                "Reuben - Corned Beef, "
                "Publix Garlic and Herb Tofu Sub, "
                "Reuben - Turkey, "
                "Publix Deli Greek Wrap, "
                "Publix Deli Meatless Turkey Club Sub Wrap Hot,"
                "Publix Ham Salad Wrap, "
                "Boar's Head Deluxe Sub, "
                "Publix Ham and Turkey Wrap, "
                "Boar's Head Deluxe Wrap, "
                "Publix Cuban Wrap, "
                "Boar's Head American Wrap, "
                "Boar's Head Low Sodium Ham Sub, "
                "Boar's Head American Sub, "
                "Publix Deli Mojo Pork Sub, "
                "Boar's Head Cajun Turkey Wrap, "
                "Boar's Head Chipotle Chicken Sub, "
                "Boar's Head Cracked Pepper Turkey Wrap, "
                "Publix Deli Meatless Philly Sub Hot."
            ),
        },
        {"role": "user", "content": f"Sandwich order: {user_order_text}"}

    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.1,
    )

    response = completion.choices[0].message['content']

    return response

