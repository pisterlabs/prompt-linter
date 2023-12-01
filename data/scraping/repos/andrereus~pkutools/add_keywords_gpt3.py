import csv
import openai
import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv("../../.env.local")
openai.api_key = os.environ["OPENAI_API_KEY"]

keywords = [
    "bayerische-brezel", "plastic-food-container", "kawaii-french-fries", "dim-sum", "kirsche", "beef", "dinner", "croissant", "dozen-eggs", "gum", "gelee", "ingredients-for-cooking", "ohne-senf", "gailan", "bananen-split", "essen-und-wein", "granatapfel", "firm-tofu", "kuerbis", "pastel-de-nata", "chocolate-bar-white", "sauce-bottle", "wenig-salz", "no-shellfish", "hamburger", "soup-plate", "sunflower-butter", "lebkuchenhaus", "hot-dog", "butter", "rice-vinegar", "vegetable-bouillion-paste", "pistachio-sauce", "proteine", "vegetarisches-essen", "onion", "mango", "fry", "roast", "stir", "bao-bun", "cuts-of-beef", "fruehlingsrolle", "steak", "ginger", "apfel", "tapas", "flour", "eier", "tempeh", "oat-milk", "cookie", "blueberry", "chili-pepper", "pfannkuchen", "dessert", "green-tea", "beet", "octopus", "healthy-food", "kawaii-ice-cream", "korean-rice-cake", "zimtschnecke", "sweets", "cereal", "black-sesame-seeds", "keine-krustentiere", "no-apple", "chard", "silken-tofu", "kawaii-soda", "salami", "salt", "heinz-bohnen", "kawaii-sushi", "coffee-capsule", "tumeric", "durian", "wine-and-glass", "potato", "kawaii-pizza", "caviar1", "nicht-vegetarisches-essen-symbol", "ohne-milch", "fischfutter", "sugar", "souvla", "flour-in-paper-packaging", "melone", "thanksgiving", "no-fish", "lime", "food-donor", "eis-in-der-waffel", "gyoza", "verfaulte-tomaten", "squash", "thyme", "potato-chips", "paella", "brezel", "vegetables-bag", "soy-sauce", "egg-basket", "zucchini", "kokosnuss", "cauliflower", "spaghetti", "deliver-food", "cake", "organic-food", "faser", "naan", "papaya", "nuss", "kekse", "brigadeiro", "pommes", "fondue", "natural-food", "pfirsich", "group-of-vegetables", "sunny-side-up-eggs", "ananas", "milchflasche", "jamon", "sellerie", "mushroom", "you-choy", "bitten-sandwich", "bok-choy", "spam-dose", "popcorn", "grains-of-rice", "coconut-milk", "macaron", "mittagessen", "milk-carton", "list-view", "karotte", "no-sugar", "broccoli", "kein-soja", "speck", "loeffel-zucker", "samosa", "keine-erdnuss", "brot", "cookbook", "kiwi", "reisschuessel", "einkaufsbeutel", "banane", "apples-plate", "rack-of-lamb", "citrus", "keine-lupinen", "radish", "taco", "black-pepper", "muschel", "rolled-oats", "cuts-of-pork", "jam", "artischocke", "hemp-milk", "peanut-butter", "spice", "hamper", "no-gmo", "mais", "cute-pumpkin", "pecan", "paprika", "brotdose", "kawaii-taco", "kohlenhydrate", "spargel", "jackfruit", "granulated-garlic", "greek-salad", "caviar", "lauch", "sosse", "breakfast", "lemonade", "haferbrei", "veganes-essen", "lettuce", "cabbage", "kawaii-bread", "zimtstangen", "flax-seeds", "bento", "eggplant", "butter-churn", "vegetarian-mark", "sandwich", "cashew", "haselnuss", "spinach", "wassermelone", "pizza", "eierkarton", "ohne-fruktose", "kawaii-cupcake", "plum", "finocchio", "kuchen", "empty-jam-jar", "kawaii-egg", "dolma", "sesame", "erdnuesse", "lipide", "no-celery", "collard-greens", "fruit-bag", "broccolini", "mcdonalds-pommes-frites", "sushi", "vegan-symbol", "salt-shaker", "himbeere", "pizza-five-eighths", "chia-seeds", "nudeln", "real-food-for-meals", "blechdose", "merry-pie", "stachelannone", "date-fruit", "grocery-shelf", "crab", "nachos", "schokoriegel", "suessstoff", "ohne-gluten", "honey-spoon", "almond-butter", "lentil", "mangosteen", "group-of-fruits", "trauben", "calories", "suessigkeit", "food-receiver", "spiess", "quesadilla", "muffin", "vegetarian-food-symbol", "curry", "no-nuts", "bagel", "zutaten", "sugar-cubes", "bread-crumbs", "no-meat", "peas", "melting-ice-cream", "erdbeere", "doughnut", "gurke", "avocado", "prawn", "garlic", "no-eggs", "tomate", "baguette", "geburtstagskuchen", "kawaii-steak", "kaese", "joghurt", "maple-syrup", "healthy-food-calories-calculator", "kohlrabi", "birne", "olivenoel", "lychee", "kawaii-coffee", "natrium", "smoked-paprika", "salat", "olive", "tea-pair", "white-beans", "zuckerwatte", "soja", "einwickeln", "bake", "raisins", "sweet-potato", "nonya-kueh", "sugar-free", "honig", "orange", "drachenfrucht", "eis-im-becher", "aprikose", "stueck-wuerfelzucker", "ohne-sesam", "lasagna", "refreshments", "wuerste", "brazil-nut", "chicken-and-waffle", "chocolate-spread"
]

def find_matching_keyword_gpt3(name):
    prompt = f"Given the following food name in German: '{name}', which keyword from the list {keywords} is the most suitable to describe it? If there is no matching keyword, choose 'organic-food'."
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=25,
        n=1,
        stop=None,
        temperature=0,
    )

    keyword = response.choices[0].text.strip()
    return keyword if keyword in keywords else "organic-food"

with open("test.csv", newline='', encoding='utf-8') as input_file, open("test-icon.csv", "w", newline='', encoding='utf-8') as output_file:
    reader = csv.reader(input_file, delimiter=',', quotechar='"')
    writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Write the header
    header = next(reader)
    header.append("keyword")
    writer.writerow(header)
    
    for row in reader:
        keyword = find_matching_keyword_gpt3(row[0])
        row.append(keyword)
        writer.writerow(row)
