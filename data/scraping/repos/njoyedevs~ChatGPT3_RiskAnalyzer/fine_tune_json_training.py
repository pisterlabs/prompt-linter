import openai
import pandas as pd
import os
from dotenv import load_dotenv
import csv
import json
import jsonlines
import random

# Load the environment variables from the .env file
load_dotenv()

# Authenticate with OpenAI API
openai.api_key = "YOUR_API_KEY"
openai.api_key = os.getenv('OPENAI_API_KEY')

def create_json_from_csv():
    
    # data = pd.read_csv("Final_Project/flask_app/models/fine_tune_example_data.csv")
    # print(data.columns)
    
    # Read CSV file
    with open('Final_Project/flask_app/models/fine_tune_example_data.csv', 'r') as f:
        data = list(csv.reader(f))
        headers = data[0]
        data = data[1:]

    # Create prompt and completion pairs
    pairs = []
    for player in data:
        prompt = f"Write a summary of {player[headers.index('Player')]}'s statistics:"
        completion = f"{player[headers.index('Player')]} played {player[headers.index('GP  Games played')]} games, starting {player[headers.index('GS  Games started')]} of them. He had an average of {player[headers.index('MPG  Minutes Per Game')]} minutes per game, scoring {player[headers.index('PPG  Points Per Game')]} points per game. He made {player[headers.index('FGM  Field Goals Made')]} out of {player[headers.index('FGA  Field Goals Attempted')]} field goals, for a field goal percentage of {player[headers.index('FG%  Field Goal Percentage')]}. He made {player[headers.index('3FGM  Three-Point Field Goals Made')]} out of {player[headers.index('3FGA  Three-Point Field Goals Attempted')]} three-point field goals, for a three-point field goal percentage of {player[headers.index('3FG%  Three-Point Field Goal Percentage')]}. He made {player[headers.index('FTM  Free Throws Made')]} out of {player[headers.index('FTA  Free Throws Attempted')]} free throws, for a free throw percentage of {player[headers.index('FT%  Free Throw Percentage')]}. He plays as {player[headers.index('Position')]} for the {player[headers.index('Team')]}."
        pairs.append({"prompt": prompt, "completion": completion})

    # Export to JSON file
    with open('Final_Project/flask_app/models/basketball.json', 'w') as f:
        json.dump(pairs, f)
        
    # Convert to JSONL File
    with open('Final_Project/flask_app/models/basketball.jsonl', 'w') as outfile:
        for d in data:
            json.dump(d, outfile)
            outfile.write('\n')
# create_json_from_csv()

def create_training_data():
    
    # data = []
    # with open('Final_Project/flask_app/models/data.csv') as f:
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         prompt = f"On {row['date']}, what is the Monthly Risk Score for Sticky Price CPI, Sticky Price CPI Less Shelter, Sticky Price CPI Less Food Energy, Sticky Price CPI Less Food Energy Shelter, Trimmed Mean PCE Inflation Rate, 16 Percent Trimmed Mean CPI, Median CPI, Flexible Price CPI, and Flexible Price CPI Less Food Energy with corresponding daily risk scores of {row['Sticky_Price_CPI_Risk_Score']},{row['Sticky_Price_CPI_Less_Shelter_Risk_Score']},{row['Sticky_Price_CPI_Less_Food_Energy_Risk_Score']},{row['Sticky_Price_CPI_Less_Food_Energy_Shelter_Risk_Score']},{row['Trimmed_Mean_PCE_Inflation_Rate_Risk_Score']},{row['16_Percent_Trimmed_Mean_CPI_Risk_Score']},{row['Median_CPI_Risk_Score']},{row['Flexible_Price_CPI_Risk_Score']},{row['Flexible_Price_CPI_Less_Food_Energy_Risk_Score']} and data points of {row['Sticky_Price_CPI']}, {row['Sticky_Price_CPI_Less_Shelter']}, {row['Sticky_Price_CPI_Less_Food_Energy']}, {row['Sticky_Price_CPI_Less_Food_Energy_Shelter']}, {row['Trimmed_Mean_PCE_Inflation_Rate']}, {row['16_Percent_Trimmed_Mean_CPI']}, {row['Median_CPI']}, {row['Flexible_Price_CPI']}, {row['Flexible_Price_CPI_Less_Food_Energy']}"
    #         completion = row['Monthly_Risk_Score']
    #         data.append({"prompt": prompt, "completion": completion})

    # with open('Final_Project/flask_app/models/prompt_completion_pairs_nj.json', 'w') as f:
    #     json.dump(data, f)
        
    with open('Final_Project/flask_app/models/data.csv') as f:
        reader = csv.DictReader(f)
        with jsonlines.open('Final_Project/flask_app/models/prompt_completion_pairs_nj.jsonl', mode='w') as writer:
            for row in reader:
                prompt = f"On {row['date']}, what is the Monthly Risk Score for Sticky Price CPI, Sticky Price CPI Less Shelter, Sticky Price CPI Less Food Energy, Sticky Price CPI Less Food Energy Shelter, Trimmed Mean PCE Inflation Rate, 16 Percent Trimmed Mean CPI, Median CPI, Flexible Price CPI, and Flexible Price CPI Less Food Energy with corresponding daily risk scores of {row['Sticky_Price_CPI_Risk_Score']},{row['Sticky_Price_CPI_Less_Shelter_Risk_Score']},{row['Sticky_Price_CPI_Less_Food_Energy_Risk_Score']},{row['Sticky_Price_CPI_Less_Food_Energy_Shelter_Risk_Score']},{row['Trimmed_Mean_PCE_Inflation_Rate_Risk_Score']},{row['16_Percent_Trimmed_Mean_CPI_Risk_Score']},{row['Median_CPI_Risk_Score']},{row['Flexible_Price_CPI_Risk_Score']},{row['Flexible_Price_CPI_Less_Food_Energy_Risk_Score']} and data points of {row['Sticky_Price_CPI']}, {row['Sticky_Price_CPI_Less_Shelter']}, {row['Sticky_Price_CPI_Less_Food_Energy']}, {row['Sticky_Price_CPI_Less_Food_Energy_Shelter']}, {row['Trimmed_Mean_PCE_Inflation_Rate']}, {row['16_Percent_Trimmed_Mean_CPI']}, {row['Median_CPI']}, {row['Flexible_Price_CPI']}, {row['Flexible_Price_CPI_Less_Food_Energy']}"
                completion = row['Monthly_Risk_Score']
                data = {"prompt": prompt, "completion": completion}
                writer.write(data)
# create_training_data()

def create_jsonl_data():

    
            
    # Define the two classes
    classes = ['Seaweed', 'Toast']

    # Define a list to store the generated examples
    examples = []

    # Define a list to store the used prompts
    used_prompts = []

    # Generate 1000 unique examples
    while len(examples) < 1000:
        # Generate a random prompt
        prompt = "What is the best food to eat in " + random.choice(["space", "lunchroom", "backyard", "ocean", "plane", "fantasy land", "October", "the end", "the sky", "a bar"])
        
        # Check if the prompt has been used before
        if prompt not in used_prompts:
            used_prompts.append(prompt)
            
            # Generate a random completion
            completion = random.choice(classes)
            
            # Create a new example and add it to the list
            example = {"prompt": prompt, "completion": completion}
            examples.append(example)

    # Output the examples as a JSONL file
    with open("Final_Project/flask_app/models/examples.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
# create_jsonl_data()

def create_random_jsonl():
    
    places = ["Grand Canyon", "Mount Everest", "Serengeti National Park", "Great Barrier Reef", "Machu Picchu", "Taj Mahal", "Stonehenge", "Niagara Falls", "Yellowstone National Park", "Angkor Wat", "Petra", "Iguazu Falls", "Victoria Falls", "Galapagos Islands", "Chichen Itza", "Sydney Opera House", "Eiffel Tower", "Great Wall of China", "Amazon Rainforest", "Hawaii Volcanoes National Park", "Colosseum", "Acropolis of Athens", "Golden Gate Bridge", "Burj Khalifa", "Pompeii", "Borobudur", "Statue of Liberty", "Terracotta Army", "Hagia Sophia", "St. Peter's Basilica", "Red Square", "Chateau de Versailles", "Kremlin", "Alhambra", "Sistine Chapel", "Sagrada Familia", "Cape of Good Hope", "Tower Bridge", "Matterhorn", "Mount Kilimanjaro", "Mount Rushmore", "Santorini", "Antarctica", "Serengeti National Park", "Kilimanjaro", "Pyramids of Giza", "Bali", "Grand Teton National Park", "Himalayas", "Yosemite National Park", "Great Smoky Mountains National Park", "Mont Saint Michel", "Plitvice Lakes National Park", "Yellowstone National Park", "Zion National Park", "Arches National Park", "Sequoia National Park", "Banff National Park", "Bryce Canyon National Park", "Joshua Tree National Park", "Glacier National Park", "Everglades National Park", "Carlsbad Caverns National Park", "Kings Canyon National Park", "Rocky Mountain National Park", "Denali National Park", "Death Valley National Park", "Kruger National Park", "Mount Everest Base Camp", "Namib Desert", "Ngorongoro Crater", "Lake Tahoe", "Lake Baikal", "Meteora", "Marrakech", "Casablanca", "Hoi An", "Phuket", "Ko Phi Phi", "Krabi", "Koh Samui", "Sukhothai Historical Park", "Boracay", "El Nido", "Bohol", "Siargao", "Puerto Princesa", "Palawan", "Coron Island", "Komodo Island", "Yogyakarta", "Borobudur Temple", "Prambanan Temple", "Seminyak" ]
    places = list(set(places))
    is_unique = len(places) == len(set(places))
    print(len(set(places)))
        
    if is_unique:
        print("All elements in the array are unique")
    else:
        print("The array contains duplicates")
    
    print(len(places))
    
    foods = ["Pizza", "Sushi", "Burger", "Tacos", "Pasta", "Steak", "Salmon", "Chicken", "Rice", "Fajitas", "Curry", "Crab", "Borscht", "Pierogi", "Kielbasa", "Tzatziki", "Spanakopita", "Dolmades", "Hummus", "Falafel", "Kebab", "Biryani", "Chaat", "Vindaloo", "Butter Chicken", "Biryani", "Naan", "Chow Mein", "Egg Rolls", "Pho", "Banh Mi", "Ramen", "Udon", "Soba", "Gimbap", "Bibimbap", "Kimchi", "Jjajangmyeon", "Katsu", "Okonomiyaki", "Takoyaki", "Soba Noodles", "Tofu", "Beef Noodle Soup", "Hot Pot", "Peking Duck", "Moo Shu Pork", "Spring Rolls", "Fish and Chips", "Shepherd's Pie", "Bangers and Mash", "Cornish Pasty", "Haggis", "Irish Stew", "Black Pudding", "Full English Breakfast", "Fish Pie", "Roast Beef", "Roast Lamb", "Toad in the Hole", "Bubble and Squeak", "Ploughman's Lunch", "Pasty", "Bakewell Tart", "Eton Mess", "Bread and Butter Pudding", "Trifle", "Scones", "Clotted Cream", "Crumpets", "Victoria Sponge Cake", "Christmas Pudding", "Sticky Toffee Pudding", "Treacle Tart", "Banoffee Pie", "Lancashire Hotpot", "Cumberland Sausage", "Stilton Cheese", "Cheddar Cheese", "Wensleydale Cheese", "Brie Cheese", "Camembert Cheese", "Raclette Cheese", "Honeycomb", "Turkish Delight", "Cannoli", "Tiramisu", "Gelato", "Panna Cotta", "Crème Brûlée", "Chocolate Fondue", "Macarons", "Croissant"]
    foods = list(set(foods))
    is_unique = len(foods) == len(set(foods))
    print(len(set(foods)))
        
    if is_unique:
        print("All elements in the array are unique")
    else:
        print("The array contains duplicates")
    
    print(len(foods))
    
    flowers  = ['Rose',  'Lily',  'Tulip',  'Daisy',  'Sunflower',  'Orchid',  'Hibiscus', 'Zinnia',  'Snapdragon',  'Crocus',  'Aster',  'Cosmos',  'Foxglove',  'Geranium',  'Jasmine',  'Poppy',  'Primrose',  'Ranunculus',  'Verbena',  'Yarrow',  'Begonia',  'Clematis',  'Delphinium',  'Fuchsia',  'Gardenia',  'Hollyhock',  'Impatiens',  'Kalanchoe',  'Lavender',  'Morning glory',  'Nasturtium',  'Osteospermum',  'Phlox',  'Queen Anne\'s lace',  'Rudbeckia',  'Salvia',  'Thistle',  'Umbrella plant',  'Viola',  'Wisteria',  'Xeranthemum',  'Yucca',  'Zantedeschia',  'Anemone',  'Bleeding heart',  'Calendula',  'Dianthus',  'Echinacea',  'Freesia',  'Gazania',  'Heather',  'Ixia',  'Jonquil',  'Kangaroo paw',  'Lobelia',  'Monarda',  'Narcissus',  'Oxalis',  'Poinsettia',  'Quince',  'Roses of Sharon',  'Saxifrage',  'Tansy',  'Uva ursi',  'Violet',  'Waxflower',  'Xanthoceras',  'Yellow flag',  'Zephyranthes',  'Aconitum',  'Bergenia',  'Camellia',  'Dendrobium',  'Erica',  'Fritillaria',  'Garden nasturtium',  'Heliotrope',  'Iberis',  'Jacaranda',  'Knapweed',  'Lewisia',  'Mimosa',  'Nemesia',  'Rafflesiaceae',  'Saffron crocus',  'Tiger lily',  'Ulex',  'Verbascum',  'Waldsteinia',  'Xyris',  'Ylang ylang',  'Zelkova']
    flowers = list(set(flowers))
    is_unique = len(flowers) == len(set(flowers))
    print(len(set(flowers)))
        
    if is_unique:
        print("All elements in the array are unique")
    else:
        print("The array contains duplicates")
    
    print(len(flowers))
    
    planets = [  "Mercury",  "Venus",  "Earth",  "Mars",  "Jupiter",  "Saturn",  "Uranus",  "Neptune",  "Pluto",  "Ceres",  "Haumea",  "Makemake",  "Eris",  "Juno",  "Vesta",  "Pallas",  "Hygiea",  "Interamnia",  "Europa", "Callisto",  "Io",  "Amalthea",  "Himalia",  "Elara",  "Pasiphae",  "Sinope",  "Lysithea",  "Ananke",  "Leda",  "Thebe",  "Adrastea",  "Metis",  "Carme",  "Pascale",  "Taygete",  "Chaldene",  "Helike",  "Kalyke",  "Iocaste",  "Erinome",  "Isonoe",  "Praxidike",  "Thyone",  "Anthe",  "Telesto",  "Calypso",  "Atlas",  "Prometheus",  "Pandora",  "Epimetheus",  "Janus",  "Dione",  "Rhea",  "Titan",  "Hyperion",  "Iapetus",  "Phoebe",  "Janus",  "Epimetheus",  "Helene",  "Telesto",  "Calypso",  "Atlas",  "Prometheus",  "Pandora",  "Hyperion",  "Iapetus",  "Phoebe",  "Pan",  "Daphnis",  "Methone",  "Anthe",  "Pallene",  "Polydeuces",  "Cassiopeia",  "Celaeno",  "Maia",  "Taygeta",  "Alcyone",  "Electra",  "Merope",  "Asterope",  "Atlas",  "Pleione",  "Hyadum I",  "Hyadum II",  "Subra",  "Scheat",  "Markab",  "Algenib",  "Algol",  "Caph",  "Ruchbah",  "Mirach",  "Almach",  "Alamak",  "Menkar",  "Mira",  "Diphda", "Betelgeuse",  "Capella",  "Deneb",  "Polaris",  "Proxima Centauri"]
    planets = list(set(planets))
    is_unique = len(planets) == len(set(planets))
    print(len(set(planets)))
        
    if is_unique:
        print("All elements in the array are unique")
    else:
        print("The array contains duplicates")
    
    print(len(planets))
    
    sequence = ['1', '2', '3', '4']
    classes = sequence * 23
    print(len(classes))
    
    
    
    list_of_prompts = []
    
    for place, food, flower, plant, classes in zip(places, foods, flowers, planets, classes):
        
        prompt_completion_pair = {"prompt": f"When you go to {place}, you will see {food}, {flower}, {plant}", "completion": f"{classes}"}
        
        list_of_prompts.append(prompt_completion_pair)
        print(prompt_completion_pair)
create_random_jsonl()




