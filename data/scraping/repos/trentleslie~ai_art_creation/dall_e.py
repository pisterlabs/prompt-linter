import openai
import requests
import datetime
import random
import csv
import re
from ai_art_creation.api.api_key import api_key
from ai_art_creation.api.chatgpt_utils import get_description, get_tags, get_title

def generate_images_from_df(prompts_df):

    # Set your OpenAI API key
    openai.api_key = api_key
    
    # Create an empty dictionary with keys "ID", "Title", "Description", "Tags", "Price", "Image Path", and "Prompt"
    img_dict = {"ID": [],
                "Target Audience": [],
                "Theme": [],
                "Style": [],
                "Elements": [],
                "Format": [],
                "Layout": [],
                "Prompt": [],
                "Title": [],
                "Description": [],
                "Tags": [],
                #"Price": [],
                "Image Path": []}
    
    # Create an empty list to store the image paths
    image_paths = []

    for index, row in prompts_df.iterrows():

        # Set the prompt
        PROMPT = row["prompt"]

        try:

            # Set prompt and output directory
            PROMPT = PROMPT + ', balanced composition'
            print(PROMPT)
            OUTPUT_DIR = "./ai_art_creation/image_processing/images_raw/"
            #OUTPUT_DIR.mkdir(exist_ok=True)

            # Call the DALL-E API
            response = openai.Image.create(
                prompt=PROMPT,
                n=1,
                size="1024x1024",
            )

            # Get the URL of the generated image
            image_url = response["data"][0]["url"]

            # Download the image
            response = requests.get(image_url)
            
            # Get the current date and time as a timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Save the image locally
            image_file = f'{OUTPUT_DIR}/{timestamp}.png'
            with open(image_file, "wb") as f:
                f.write(response.content)
            
            # Add the image path to the list
            image_paths.append(image_file)

            # Define a regular expression pattern to match aspect ratio terms
            #aspect_ratio_pattern = re.compile(r"\b(?:aspect|ratio|1:1|4:3|3:2|16:9|2:1|wall|tshirt)\b", re.IGNORECASE)

            # Add the image to the dictionary
            img_dict["ID"].append(timestamp)
            img_dict["Target Audience"].append(row["target_audience"])
            img_dict["Theme"].append(row["theme"])
            img_dict["Style"].append(row["style"])
            img_dict["Elements"].append(row["elements"])
            img_dict["Format"].append(row["format"])
            img_dict["Layout"].append(row["layout"])
            img_dict["Title"].append(row["title"])
            img_dict["Description"].append(row["description"])
            img_dict["Tags"].append(row["tags"])
            #img_dict["Price"].append(round(random.uniform(2.00, 8.00), 2))
            img_dict["Image Path"].append(r'C:\Users\trent\OneDrive\Documents\GitHub\ai_art_creation\ai_art_creation\image_processing\images_processed' + '\\' + timestamp + '.png')
            img_dict["Prompt"].append(PROMPT)

            print(f"Image saved to: {image_file}")
            
        except Exception as e:
            print(e)
            continue
    
    # Write the dictionary to a CSV file
    csv_file_name = f'./ai_art_creation/image_processing/csv/img_database_{timestamp}.csv'
    with open(csv_file_name, "w", newline="", encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(img_dict.keys())
        for row in zip(*img_dict.values()):
            writer.writerow(row)
        
    # Return the list of image paths
    return image_paths

def generate_images(prompts):

    # Set your OpenAI API key
    openai.api_key = api_key
    
    # Create an empty dictionary with keys "ID", "Title", "Description", "Tags", "Price", "Image Path", and "Prompt"
    img_dict = {"ID": [],
                "Title": [],
                "Description": [],
                "Tags": [],
                #"Price": [],
                "Image Path": [],
                "Prompt": []}
    
    # List of 100 unique objects
    objects = []
    while len(objects) < 100:
        # Generate a random word using a list of adjectives, animals, and verbs
        adj = ['Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange', 'Black', 'White', 'Pink', 'Gray', 'Brown', 'Silver', 'Pink', 'Gray', 'Brown', 'Silver', 'Rainbow']
        adj2 = ['Pixar style 3D render', 'Bitter', 'Smooth', 'Brave', 'Clever', 'Loud', 'Gentle', 'Silly', 'Sour', 'Shy', 'Wise', 'Fierce', 'Lively', 'Proud', 'Calm', 'Angry', 'Happy', 'Sad', 'Stupid', 'Witty', 'Mysterious', 'Fat', 'Skinny', 'Big', 'Small']
        animal = ['Octopus' , 'Elephant', 'Giraffe', 'Hippo', 'Monkey', 'Kangaroo', 'Tiger', 'Bear', 'Zebra', 'Rhino', 'Crocodile', 'Penguin', 'Ostrich', 'Koala', 'Panda', 'Gorilla', 'Camel', 'Hedgehog', 'Squirrel', 'Fox'
                                ]
        nouns = ['Apple', 'Banana', 'Car', 'Dog', 'Elephant', 'Fish', 'Guitar', 'House', 'Ice Cream', 'Jacket', 'Kangaroo', 'Lion', 'Mountain', 'Nose', 'Orange', 'Pizza', 'Queen', 'Robot', 'Shoe', 'Tree', 'Umbrella', 'Violin', 'Waterfall', 'Xylophone', 'Yacht', 'Zebra', 'Airplane', 'Ball', 'Camera', 'Desk', 'Egg', 'Flower', 'Garden', 'Hat', 'Island', 'Juice', 'Key', 'Laptop', 'Moon', 'Notebook', 'Ocean', 'Pencil', 'Ring', 'Sun', 'Table', 'Unicorn', 'Volcano', 'Whale', 'Yogurt']
        verb = ['Run', 'Jump', 'Swim', 'Fly', 'Climb', 'Sleep', 'Eat', 'Sing', 'Dance', 'Play']
        word = random.choice(verb) + 'ing ' + random.choice(adj) + ' ' + random.choice(adj2) + ' ' + random.choice(animal)
        word2 = random.choice(adj) + ' ' + random.choice(adj2) + ' ' + random.choice(nouns)
    
        # Add the word to the list if it's not already in there
        if word not in objects:
            #print(word)
            objects.append(word)
            
        # Add the word to the list if it's not already in there
        if word2 not in objects:
            objects.append(word2)
    
    # add 100 empty strings to the end of the list
    objects += [""] * 100
    #objects += ["irregular white #FFFFFF border"] * 20
    
    # Create an empty list to store the image paths
    image_paths = []
    
    # Create a list of prompts and append the forced prompts
    forced_prompts = [
                    "An original, creative, trendy, fun pattern with " + random.choice(objects) + ", suitable for printing on a t-shirt or mug, without depicting an acutal t-shirt or mug. Again, do NOT include an actual t-shirt or mug in the image.",
                    #"Abstract geometric patterns with vibrant colors with watercolor textures",
                    #"Elegant floral designs with watercolor textures",
                    #"Galaxy and celestial elements with glowing stars with watercolor textures",
                    #"Modern reinterpretation of classic works of art with a digital twist"
                    #"Imagine a piece of art that captures the serene beauty of the first snowfall. This mixed media masterpiece uses a unique combination of watercolors and delicate line drawings to depict snowflakes dancing their way down to the ground, illuminated by soft moonlight or the glow of streetlights. The artist has expertly blended these elements to create a stunning composition that would make an incredible addition to any winter space. Perfect for your desktop wallpaper, this piece will transport you to a peaceful winter wonderland where you can enjoy the tranquility of nature's gentlest moments.",
                    #"Imagine a piece of art that captures the serene beauty of the spring rain. This mixed media masterpiece uses a unique combination of watercolors and delicate line drawings to depict spring flowers bursting their from down to the ground, illuminated by soft moonlight or the glow of streetlights. The artist has expertly blended these elements to create a stunning composition that would make an incredible addition to any spring space. Perfect for your desktop wallpaper, this piece will transport you to a peaceful spring wonderland where you can enjoy the tranquility of nature's gentlest moments.",
                    #"Imagine a piece of art that captures the serene beauty of the autumn forest. This mixed media masterpiece uses a unique combination of watercolors and delicate line drawings to depict leaves dancing their way down to the ground, illuminated by soft moonlight or the glow of streetlights. The artist has expertly blended these elements to create a stunning composition that would make an incredible addition to any autumn space. Perfect for your desktop wallpaper, this piece will transport you to a peaceful autumn wonderland where you can enjoy the tranquility of nature's gentlest moments.",
                    #"Psychedelic swirls of vivid colors intertwining with intricate fractal patterns with watercolor textures",
                    #"A dreamlike tropical oasis filled with lush plants, exotic birds, and shimmering waterfalls with watercolor textures",
                    #"A cosmic dance of celestial bodies, swirling galaxies, and vibrant nebulas with watercolor textures",
                    #"Graffiti-style street art incorporating elements of technology and AI in a vibrant urban setting with watercolor textures",
                    #"Vintage-inspired botanical illustrations with delicate linework and pastel hues with watercolor textures",
                    #"Abstract interpretation of sound waves and musical notes in a harmonious color palette with watercolor textures",
                    #"A surreal dreamscape merging elements of nature, technology, and human imagination with watercolor textures"
                ] * 10
    prompts.extend(forced_prompts)

    for PROMPT in prompts:

        try:

            modifiers = 'balanced composition, ' + random.choice(objects)  

            # Set prompt and output directory
            PROMPT = PROMPT + ', ' + modifiers
            print(PROMPT)
            OUTPUT_DIR = "./ai_art_creation/image_processing/images_raw/"
            #OUTPUT_DIR.mkdir(exist_ok=True)

            # Call the DALL-E API
            response = openai.Image.create(
                prompt=PROMPT,
                n=1,
                size="1024x1024",
            )

            # Get the URL of the generated image
            image_url = response["data"][0]["url"]

            # Download the image
            response = requests.get(image_url)
            
            # Get the current date and time as a timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Save the image locally
            image_file = f'{OUTPUT_DIR}/{timestamp}.png'
            with open(image_file, "wb") as f:
                f.write(response.content)
            
            # Add the image path to the list
            image_paths.append(image_file)

            # Define a regular expression pattern to match aspect ratio terms
            aspect_ratio_pattern = re.compile(r"\b(?:aspect|ratio|1:1|4:3|3:2|16:9|2:1|wall|tshirt)\b", re.IGNORECASE)

            # Add the image to the dictionary
            img_dict["ID"].append(timestamp)
            img_dict["Title"].append(get_title(PROMPT).replace('"', ''))
            img_dict["Description"].append(get_description(aspect_ratio_pattern.sub("", PROMPT)))
            img_dict["Tags"].append(get_tags(aspect_ratio_pattern.sub("", PROMPT)).replace("\n", ", ").replace("[0-9]", "").replace("[0-9]\.", ""))
            #img_dict["Price"].append(round(random.uniform(2.00, 8.00), 2))
            img_dict["Image Path"].append(r'C:\Users\trent\OneDrive\Documents\GitHub\ai_art_creation\ai_art_creation\image_processing\images_processed' + '\\' + timestamp + '.png')
            img_dict["Prompt"].append(PROMPT)

            print(f"Image saved to: {image_file}")
            
        except Exception as e:
            print(e)
            continue
    
    # Write the dictionary to a CSV file
    csv_file_name = f'./ai_art_creation/image_processing/csv/img_database_{timestamp}.csv'
    with open(csv_file_name, "w", newline="", encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(img_dict.keys())
        for row in zip(*img_dict.values()):
            writer.writerow(row)
        
    # Return the list of image paths
    return image_paths