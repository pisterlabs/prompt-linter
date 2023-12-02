import csv
import os
import random
import requests

from django.core.management.base import BaseCommand
from openai import OpenAI

from api.models import Ingredient
from api.models import Recipe


def download_image(url, recipeName):
    response = requests.get(url, stream=True)
    image_name = recipeName.replace(" ", "_").lower() + ".jpg"
    with open("./static/" + image_name, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

    return image_name


def create_prompt(arg):
    client = OpenAI(
        api_key="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",
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Generate a short,funny description about " + arg
            }
        ],
        model="gpt-3.5-turbo",
    )

    return response.choices[0].message.content


def create_image(arg):
    client = OpenAI(
        api_key="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",
    )

    response = client.images.generate(
        model="dall-e-3",
        prompt=arg,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    return image_url


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('arg1', type=str, help='Path to a file')

    def handle(self, *args, **options):
        recipeNames = []
        recipeIngredients = []

        input_csv_path = options['arg1']

        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f'The file "{input_csv_path}" does not exist.')

        with open(input_csv_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)

            for row in csv_reader:
                if (len(row) > 0):
                    recipeNames.append(row[0])
                    recipeIngredients.append(row[1:])

        recipeIndex = 0
        recipeId = 0
        ingredientId = len(recipeNames)



        for recipeName in recipeNames:
            randomDifficulty = random.randint(1, 6)
            randomDuration = random.randint(10, 121)
            associatedDescription = create_prompt(recipeName)
            print(associatedDescription)
            imageUrl = create_image(recipeName)
            print(imageUrl)
            new_image_url = download_image(imageUrl, recipeName)
            recipe = Recipe(id=recipeId, name=recipeName, description=associatedDescription,
                            difficulty=randomDifficulty, duration=randomDuration, image_id=new_image_url)
            recipeId+=1
            recipe.save()
            for ingredientName in recipeIngredients[recipeIndex]:
                if (not Ingredient.objects.filter(name=ingredientName).exists()):
                    ingredient = Ingredient(id=ingredientId, name=ingredientName)

                    ingredient.save()
                    ingredientId+=1
                    recipe.ingredients.add(ingredient)

                else:
                    print("Duplicate Ingredient detected")
                    existingIngredientEntity = Ingredient.objects.get(name=ingredientName)
                    recipe.add(existingIngredientEntity)

            recipeIndex += 1

        self.stdout.write(self.style.SUCCESS('Successfully generated recipes database and AI images'))
