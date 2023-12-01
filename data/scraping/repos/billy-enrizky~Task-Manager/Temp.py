import pytesseract
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

def recommend_menu(image_url, restriction):
    # URL of the image
    #image_url = "https://firebasestorage.googleapis.com/v0/b/developmentdatabase-b3fbc.appspot.com/o/images%2Fimage_4.jpg?alt=media&token=f07cc9d2-4571-4af4-8570-d0a7ce2d6618&_gl=1*oibvqh*_ga*OTg1NjMxMjQ1LjE2OTM2Mjc1MzY.*_ga_CW55HF8NVT*MTY5NzMwNzY1Ny45NS4xLjE2OTczMDkxNzguNjAuMC4w"
    
    # Download the image from the URL
    response = requests.get(image_url)
    
    from io import BytesIO
    img = Image.open(BytesIO(response.content))
    # Perform OCR
    text = pytesseract.image_to_string(img)
    
    # Print or use the extracted text
    
    import openai
    
    # Set your OpenAI API key
    api_key = "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"
    
    # Your input prompt to generate a list of ingredients for a food dish
    prompt = f"find all foods from the text: {text}"
    
    # Make a request to GPT-3
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose the appropriate engine
        prompt=prompt,
        max_tokens=1000,  # Adjust as needed
        temperature=0.0,
        api_key=api_key
    )
    
    # Extract the generated ingredients list from the response
    ingredients = response.choices[0].text
    ingredients = ingredients[ingredients.find(':') + 1:].strip()
    
    
    import requests
    
    # Your Edamam API credentials
    app_id = '34340b2c'
    app_key = 'aa54ba6e5ea8ffdd204444757a4983d2'
    
    def find_ingredients(dish_name):
        # Make the API request
        url = f'https://api.edamam.com/api/food-database/parser'
        params = {
            'app_id': app_id,
            'app_key': app_key,
            'ingr': dish_name,
        }
        response = requests.get(url, params=params)
        data = response.json()
        # Initialize the index variable to None
        index_with_food_content_label = None
        # Iterate through the list
        for i, item in enumerate(data['hints']):
            if 'foodContentsLabel' in item['food']:
                index_with_food_content_label = i
                break    
        ingredients_string = data['hints'][index_with_food_content_label]['food']['foodContentsLabel'].split(';')
        return ingredients_string
    
    dish_n_ingredients = []
    dishes = ingredients.strip().split(',')
    
    for dish in dishes:
        ingredients = find_ingredients(dish)
        dish_n_ingredients.append({'Dish': dish, 'Ingredients': ingredients})
    
    # Create a DataFrame from the list of dish-ingredient pairs
    df = pd.DataFrame(dish_n_ingredients)
    #print("Tell us Your Restriction (Separate each item using comma)")
    #restriction = input("Enter Restriction: ")
    restrictions = restriction.split(',')
    # Convert the restriction list to lowercase for case-insensitive comparison
    restrictions = [item.lower() for item in restrictions]
    
    # Filter the DataFrame to remove rows where any item in 'restriction' is in the list of ingredients
    df = df[~df['Ingredients'].apply(lambda ingredients: any( restriction in ingredient.lower() for ingredient in ingredients for restriction in restrictions))]
    
    # Reset the index after removing rows
    df.reset_index(drop=True, inplace=True)
    
    json_data = json.dumps(df['Dish'].to_list())
    
    return json_data

