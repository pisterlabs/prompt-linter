import os
import openai
import urllib.request

from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost
from wordpress_xmlrpc.compat import xmlrpc_client
from wordpress_xmlrpc.methods import media, posts

import random
from twilio.rest import Client as twilioClient
import time
from datetime import date

import yaml
import shutil

import requests

def load_credentials(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials

credentials_file = "LisaCredentials.yaml"
credentials = load_credentials(credentials_file)


openai.api_key = credentials['OPENAI_API_KEY']
twilioID = credentials['TWILIO_ID']
twilioToken = credentials['TWILIO_TOKEN']
numberFrom = credentials['LISA_NUMBER']
numberTo= credentials['CONFIRMATION_NUMBER']
instaUsername = credentials['INSTAGRAM_USERNAME']
instaPassword = credentials['INSTAGRAM_PASSWORD']
access_token = credentials['INSTAGRAM_LONG_TOKEN']

class Lisa:
    
    def __init__(self, bio):
        
        self.Bio = bio

        self.History = self.Bio+"\n\n"
        
        self.mostRecentRecipes = []
        
        self.mostRecentImage = ""
        
    def getResponse(self, prompted):
        
        promptFull = self.History+prompted
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            #prompt=promptFull,
            messages=[
                {"role": "system", "content": self.History},
                {"role": "user", "content": prompted},
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
            
        responseText = response.choices[0].message.content
        
        self.History = promptFull+responseText+"\n\n"
        
        self.mostRecentRecipes.append(responseText)
            
        return responseText
    
    def save(self):
        LisaHistory = open("Lisa's History.txt", "w")
        n = LisaHistory.write(self.History)
        LisaHistory.close()
        
    def load(self):
        with open("Lisa's History.txt", 'r') as file: 
            LisaHistory = file.read()

        self.History = LisaHistory
        
    def getImage(self, imagePrompt):
        
        
        response = openai.Image.create(
          prompt = imagePrompt+" Warm and inviting lighting, visually appealing and appetizing presentation. Food blog photo, 1080p, photorealistic, warm colors. ",
          n=1,
          size="1024x1024"
        )
        image_url = response['data'][0]['url']
        
        self.mostRecentImage = image_url
        
        urllib.request.urlretrieve(self.mostRecentImage, "images/recipe.jpg")
        
        
    def makeInstagramPost(self, caption):
        
        image_url = self.mostRecentImage
        
        url = f"https://graph.facebook.com/v13.0/me/media?access_token={access_token}"
        data = {
            "image_url": image_url,
            "caption": caption
        }

        # Step 1: Create a media object
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print("Error creating media object:", response.text)
            return

        media_id = response.json().get("id")
        if not media_id:
            print("Media ID not found in response:", response.text)
            return

        # Step 2: Publish the media object
        url = f"https://graph.facebook.com/v13.0/{media_id}/publish?access_token={access_token}"
        response = requests.post(url)
        if response.status_code != 200:
            print("Error publishing media object:", response.text)
            return

        print("Media published successfully:", response.json())
        
        
    def generateAndUploadPost(self, mealType='dessert', hints=''):
        
        mealPrompts = {
            'breakfast': "Write me a blog post for a breakfast recipe!",
            'lunch': "Write me a blog post about an easy lunch recipe that you'd make for yourself in 15 minutes. Only use what you have in the fridge!",
            'dinner': "Write me a blog post for an easy dinner recipe!",
            'dessert': "Write me a blog post for a great dessert recipe that you like to make for your kids!"
        }

        categories = {'breakfast': 'Breakfast', 'lunch': 'Lunches', 'dinner': 'Easy Dinners', 'dessert': 'Desserts'}

        ingredients = ['Salt',
                         'Pepper',
                         'Sugar',
                         'Flour',
                         'Butter',
                         'Olive oil',
                         'Garlic',
                         'Onions',
                         'Eggs',
                         'Milk',
                         'Cream',
                         'Cheese',
                         'Bread',
                         'Rice',
                         'Noodles',
                         'Vegetable oil',
                         'Broth',
                         'Tomatoes',
                         'Spinach',
                         'Carrots',
                         'Celery',
                         'Potatoes',
                         'Chicken',
                         'Beef',
                         'Pork',
                         'Fish',
                         'Beans',
                         'Herbs',
                         'Spices',
                         'Vanilla',
                         'Baking powder',
                         'Baking soda',
                         'Yeast',
                         'Nuts',
                         'Chocolate',
                         'Honey',
                         'Maple syrup',
                         'Apples',
                         'Bananas',
                         'Strawberries',
                         'Lemons',
                         'Limes',
                         #'Broccoli',
                         'Bell peppers',
                         'Mushrooms',
                         'Basil',
                         'Oregano',
                         'Parsley',
                         'Cilantro',
                         'Paprika',
                         'Cayenne pepper',
                         'Mustard',
                         'Mayonnaise',
                         'Ketchup',
                         'Worcestershire sauce',
                         'Vinegar',
                         'Soy sauce',
                         'Barbecue sauce',
                         'Hot sauce',
                         'Coconut milk',
                         'Almonds',
                         'Walnuts',
                         'Avocado',
                         'Sunflower seeds',
                         'Sesame seeds',
                         'Raisins',
                         'Dried cranberries',
                         'Olives',
                         'Cashews',
                         'Cornstarch',
                         'Millet',
                         'Quinoa',
                         'Rye',
                         'Oats',
                         'Pecans',
                         'Garlic powder',
                         'Onion powder',
                         'Cinnamon',
                         'Nutmeg',
                         'Allspice',
                         'Cloves',
                         'Ginger',
                         'Coconut',
                         'Artichoke',
                         'Green beans',
                         'Zucchini',
                         'Peas',
                         'Asparagus',
                         'Cucumber',
                         'Plum',
                         'Dates',
                         'Peaches',
                         'Figs',
                         'Cream cheese',
                         'Ricotta cheese',
                         'Anchovies',
                         'Salmon',
                         'Squid',
                         'Tofu',
                         'Seaweed']

        posted=False


        print("Generating recipe...")
        prompt = mealPrompts[mealType]+' Only provide me the text, not the title, but write the recipe exactly as a mom would.'
        print(f"Meal type: {mealType}")
        if hints != '' and hints != None: 
            prompt += " Use the following as hints, but only if they suit the meal type: "+hints+"."
        elif hints==None:
            pass
        else:
            ingredient1 = random.choice(ingredients)
            ingredient2 = random.choice(ingredients)
            ingredient3 = random.choice(ingredients)
            print(f"Ingredients used: {ingredient1}, {ingredient2}")
            prompt += f" If you want to, use one of the following ingredients as hints, but only if they suit the meal type: {ingredient1}, {ingredient2}, {ingredient3}."
        prompt += f" The current month is {date.today().strftime('%B')}. "
        postContent = self.getResponse(prompt)

        print(self.mostRecentRecipes[0])

        print("getting Actual meal category...")
        actualMealType = self.getResponse("\n\nIn one word, between the options breakfast, lunch, dinner, and dessert, in lowercase and with no punctuation, what type of meal is the above recipe?")
        print(actualMealType)
        mealType=actualMealType

        print("Getting post title...")
        postTitle = self.getResponse("\n\nWhat's a SEO optimized title for the above post? Make sure you include the recipe name.")
        print(postTitle)

        print("Getting image prompt...")

        imagePrompt = self.getResponse(f"\n\nWhat would be a good prompt for DALL-E to generate a thumbnail for the '{postTitle}' recipe? Include as much detail as possible about the ingredients found in the recipe. Focus on the food. Be very straightforward and don't include anything personal or the word recipe.")
        print(imagePrompt)

        print("Generating image...")
        self.getImage(imagePrompt)

        print("Generating Instagram caption...")
        instaCaption = self.getResponse(f"\n\nWrite me an Instagram post for this recipe. The only photo in the post will be the image you just generated. Make sure you provide a link to lisainthekitchen.com since that's where the recipe will be.")
        print(instaCaption)

        print("Image:")
        print(self.mostRecentImage)
        print("Verify that you want to post this recipe (y/n):")

        client = twilioClient(twilioID, twilioToken)
        message = client.messages.create(
            to=numberTo, 
            from_=numberFrom,
            body=f"Do you want to post this recipe? Title: {postTitle}. Category: {mealType}. Text back 'Y' to confirm.",
            media_url=self.mostRecentImage)

        #yesorno = input()

        yesorno = 'n'
        startTime = time.time()
        while yesorno == 'n': 

            mostRecent = client.messages.list()[0].body
            #mostRecent1 = client.messages.list()[1].body

            if mostRecent in ['Y', 'y', 'Yes', 'yes']:
                yesorno = 'y'
            elif mostRecent in ['N', 'n', 'No', 'no']:
                yesorno='break'
            if time.time()-startTime>300:
                print("Waited too long.")
                yesorno='break'
            time.sleep(1)

            #print(mostRecent)

        if yesorno=="y":
            posted=True


        if posted:

            # bunch of wordpress stuff to fetch image
            wp = Client('https://lisainthekitchen.com/xmlrpc.php', 'maxmarcussen98', '$Alazar98')

            # set to the path to your file
            imageFilename = 'images/recipe.jpg'

            # prepare metadata
            data = {
                    'name': 'recipe.jpg',
                    'type': 'image/jpeg',  # mimetype
            }

            # read the binary file and let the XMLRPC library encode it into base64
            with open(imageFilename, 'rb') as img:
                    data['bits'] = xmlrpc_client.Binary(img.read())

            response = wp.call(media.UploadFile(data))
            # response == {
            #       'id': 6,
            #       'file': 'picture.jpg'
            #       'url': 'http://www.example.com/wp-content/uploads/2012/04/16/picture.jpg',
            #       'type': 'image/jpeg',
            # }
            attachment_id = response['id']

            # all of the stuff to make post
            print("Posting to WordPress...")
            post = WordPressPost()
            titlePost = postTitle.lstrip()[0:postTitle.lstrip().find('\n')].strip('\"')
            post.title = titlePost
            post.terms_names = {'category': [categories[mealType]]}
            post.content = postContent.lstrip()[:]
            post.thumbnail = attachment_id
            post.id = wp.call(NewPost(post))
            post.post_status = 'publish'
            wp.call(posts.EditPost(post.id, post))

            print("Done")

            client.messages.create(
                to=numberTo, 
                from_=numberFrom,
                body=f"Recipe posted! Please go to lisainthekitchen.com to confirm",
            )

            print("Posting to instagram...")
            self.makeInstagramPost(caption=instaCaption)

        else:
            print("Try again.")

            client.messages.create(
                to=numberTo, 
                from_=numberFrom,
                body=f"Recipe not posted.",
            )

        
    def autoPost(self):
        
        postTypes = ['breakfast', 'lunch', 'dinner', 'dessert']
        
        self.generateAndUploadPost(random.choice(postTypes))
        
        
        
        
if __name__=="__main__":
    with open("Lisa's History.txt", 'r') as file: 
        LisaHistory = file.read()

    LisaNew = Lisa(LisaHistory)

    LisaNew.autoPost()
        