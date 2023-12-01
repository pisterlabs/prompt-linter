import os
from dotenv.main import load_dotenv
from langchain.llms import OpenAI
#from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import easyocr
import cv2
from streamlit_option_menu import option_menu

load_dotenv()
llm = OpenAI(openai_api_key=os.environ["KEY"])

def read_img(img):
    try:
        image = Image.open(img)
        reader = easyocr.Reader(['en']) 
        text = reader.readtext(image, detail=0)
        return(str(text))
    except Exception as e:
        return "An error has occurred."
    
def plan(s, g, t):
    temp = """Limit your response to 100 words. Given a start weight, goal weight, and time span in months, create a meal plan that will help the user reach their goal weight in the given time span."

    start: 200
    goal: 160
    time: 6
    Answer: Start weight: 200 lbs
    End weight: 160 lbs
    Timeframe: 6 months
    Calories: 1500-1600 calories per day
    Macronutrients: 40% protein, 30% carbohydrates, 30% fat

    Breakfast

    Oatmeal with berries and nuts (300 calories)
    Greek yogurt with fruit and granola (300 calories)
    Eggs with whole-wheat toast and avocado (300 calories)
    Lunch

    Salad with grilled chicken or fish (400 calories)
    Soup and sandwich on whole-wheat bread (400 calories)
    Leftovers from dinner (400 calories)
    Dinner

    Grilled salmon with roasted vegetables (400 calories)
    Chicken stir-fry with brown rice (400 calories)
    Lentil soup with whole-wheat bread (400 calories)
    Snacks

    Fruits and vegetables
    Nuts and seeds
    Hard-boiled eggs
    Greek yogurt
    Total calories: 1500-1600 calories
    Total protein: 120-128 grams
    Total carbohydrates: 135-144 grams
    Total fat: 50-56 grams
    
    
    start: 180
    goal: 170
    time: 9
    Answer: Start weight: 180 lbs
    End weight: 170 lbs
    Timeframe: 9 months
    Calories: 1700-1800 calories per day
    Macronutrients: 40% protein, 30% carbohydrates, 30% fat

    Breakfast

    Oatmeal with berries and nuts (350 calories)
    Greek yogurt with fruit and granola (350 calories)
    Eggs with whole-wheat toast and avocado (350 calories)
    Lunch

    Salad with grilled chicken or fish (450 calories)
    Soup and sandwich on whole-wheat bread (450 calories)
    Leftovers from dinner (450 calories)
    Dinner

    Grilled salmon with roasted vegetables (450 calories)
    Chicken stir-fry with brown rice (450 calories)
    Lentil soup with whole-wheat bread (450 calories)
    Snacks

    Fruits and vegetables
    Nuts and seeds
    Hard-boiled eggs
    Greek yogurt
    Total calories: 1700-1800 calories
    Total protein: 136-144 grams
    Total carbohydrates: 152-160 grams
    Total fat: 64-72 grams

    This meal plan provides a variety of nutritious foods from all food groups.
    

    start: 180
    goal: 190
    time: 2
    Answer: Sample Meal Plan for Weight Gain

    Start weight: 180 lbs
    End weight: 190 lbs
    Timeframe: 2 months
    Calories: 2500-2600 calories per day
    Macronutrients: 40% protein, 30% carbohydrates, 30% fat

    Breakfast

    Oatmeal with berries, nuts, and nut butter (500 calories)
    Greek yogurt with fruit, granola, and seeds (500 calories)
    Eggs with whole-wheat toast, avocado, and cheese (500 calories)
    Lunch

    Salad with grilled chicken or fish, quinoa, and avocado (600 calories)
    Sandwich on whole-wheat bread with lean protein, cheese, and vegetables (600 calories)
    Leftovers from dinner (600 calories)
    Dinner

    Salmon with roasted vegetables and brown rice (600 calories)
    Chicken stir-fry with brown rice and nuts (600 calories)
    Lentil soup with whole-wheat bread and avocado (600 calories)
    Snacks

    Fruits and vegetables
    Nuts and seeds
    Hard-boiled eggs
    Greek yogurt
    Protein shakes
    Total calories: 2500-2600 calories
    Total protein: 200-208 grams
    Total carbohydrates: 225-234 grams
    Total fat: 83-87 grams

    This meal plan provides a variety of nutritious foods from all food groups.
    

    start: {start}
    goal: {goal}
    time: {time}
    Answer:"""
    prompt_template = PromptTemplate(
        input_variables=["start", "goal", "time"],
        template= temp
    )
    return llm(prompt_template.format(start = s, goal = g, time=t))
    
def macro(macronutrient):
    if macronutrient == "Fats":
        macronutrient = 'Lipid'
    temp = """Limit your response to 100 words. Answer the question based on the context below. You are a helpful health and cooking assistant. Based on the Macronutrient given, create a recipe centered around that macronutrient.
    
    macronutrient: protein

    Answer: Grilled Chicken Quinoa Bowl

    Ingredients:

    Chicken breasts
    Olive oil, garlic powder, paprika
    Quinoa, broth, lemon, parsley
    Mixed greens, cherry tomatoes, cucumber, red onion
    Feta (optional), hummus, lemon wedges
    Instructions:

    Marinate and grill chicken.
    Cook quinoa, add lemon, parsley.
    Assemble bowls with quinoa, grilled chicken, greens, tomatoes, cucumber, onion. Add feta if desired.
    Serve with hummus and lemon wedges.
    This protein-packed Grilled Chicken Quinoa Bowl makes a balanced, flavorful meal. Enjoy!


    macronutrient: carbohydrates

    Answer: Mediterranean Quinoa Salad

    Ingredients:

    1 cup quinoa, cooked and cooled
    1 cup cherry tomatoes, halved
    1 cucumber, diced
    1/2 red onion, finely chopped
    1/4 cup Kalamata olives, pitted and sliced
    1/4 cup crumbled feta cheese
    1/4 cup fresh parsley, chopped
    For the Dressing:

    3 tablespoons extra-virgin olive oil
    2 tablespoons lemon juice
    1 garlic clove, minced
    Salt and pepper to taste
    Instructions:

    Combine all salad ingredients in a bowl.
    In a separate bowl, whisk together the dressing ingredients.
    Drizzle the dressing over the salad, toss, and serve. Enjoy!


    macronutrient: lipid

    Answer: Avocado and Smoked Salmon Toast

    Ingredients:

    2 slices whole-grain bread
    1 ripe avocado
    4 oz smoked salmon
    1 small red onion, thinly sliced
    1 lemon, sliced into wedges
    Fresh dill for garnish
    Salt and pepper to taste
    Instructions:

    Toast the bread.
    Mash the avocado and spread it on the toasted slices.
    Top with smoked salmon, red onion slices, and a squeeze of lemon juice.
    Season with salt and pepper, garnish with fresh dill, and enjoy this lipid-rich, omega-3-packed meal!

    
    macronutrient: {macro}

    Answer: """
    prompt_template = PromptTemplate(
        input_variables=["macro"],
        template= temp
    )
    return llm(prompt_template.format(macro = macronutrient))

def allergy(allergy, text):
    temp = """Answer the question based on the context below. If there are no matches between allergies and allergens in the food, write "No restrictions were detected.".

    Context: Food allergies are hard to maintain and find, based on the user given allergies, see if there are any in the food. 


    Allergies: Eggs

    Food: ['Ingredients: Enriched Corn', 'Meal (Corn Meal,', 'Ferrous Sulfate_', 'Niacin,  Thiamin', 'Mononitrate', 'Riboflavin; Folic Acid) , Vegetable Oil (Corn, Canola;', 'and/or Sunflower Oil) , Cheese Seasoning (Whey,', 'Cheddar Cheese [Milk; Cheese Cultures, Salt,', 'Enzymes]', 'Canola Oil,', 'Maltodextrin [Made from', 'Corn],', 'Natural and Artificial Flavors,', 'Salt; Whey', 'Protein  Concentrate ,', 'Monosodium', 'Glutamate', '1', 'Lactic Acid, Citric Acid, Artificial Color [Yellow 6]) ,', 'and Salt ']

    Answer: No restrictions were detected.


    Allergies: Milk

    Food: ['Ingredients: Enriched Corn', 'Meal (Corn Meal,', 'Ferrous Sulfate_', 'Niacin, Thiamin', 'Mononitrate', 'Riboflavin; Folic Acid) , Vegetable Oil (Corn, Canola;', 'and/or Sunflower Oil) , Cheese Seasoning (Whey,', 'Cheddar Cheese [Milk; Cheese Cultures, Salt,', 'Enzymes]', 'Canola Oil,', 'Maltodextrin [Made from', 'Corn],', 'Natural and Artificial Flavors,', 'Salt; Whey', 'Protein Concentrate ,', 'Monosodium', 'Glutamate', '1', 'Lactic Acid, Citric Acid, Artificial Color [Yellow 6]) ,', 'and Salt ']

    Answer: This contains Milk, which was listed in your allergies.


    Allergies: Milk

    Food: ['Ingredients: Enriched Corn', 'Meal (Corn Meal,', 'Ferrous Sulfate_', 'Niacin, Thiamin', 'Mononitrate', 'Riboflavin; Folic Acid) , Vegetable Oil (Corn, Canola;', 'and/or Sunflower Oil) , Cheese Seasoning (Whey,', 'Cheddar Cheese [Milk; Cheese Cultures, Salt,', 'Enzymes]', 'Canola Oil,', 'Maltodextrin [Made from', 'Corn],', 'Natural and Artificial Flavors,', 'Salt; Whey', 'Protein Concentrate ,', 'Monosodium', 'Glutamate', '1', 'Lactic Acid, Citric Acid, Artificial Color [Yellow 6]) ,', 'and Salt ']

    Answer: This contains Milk, which was listed in your allergies.


    Allergies: None

    Food: ['Ingredients: Enriched Corn', 'Meal (Corn Meal,', 'Ferrous Sulfate_', 'Niacin,  Thiamin', 'Mononitrate', 'Riboflavin; Folic Acid) , Vegetable Oil (Corn, Canola;', 'and/or Sunflower Oil) , Cheese Seasoning (Whey,', 'Cheddar Cheese [Milk; Cheese Cultures, Salt,', 'Enzymes]', 'Canola Oil,', 'Maltodextrin [Made from', 'Corn],', 'Natural and Artificial Flavors,', 'Salt; Whey', 'Protein  Concentrate ,', 'Monosodium', 'Glutamate', '1', 'Lactic Acid, Citric Acid, Artificial Color [Yellow 6]) ,', 'and Salt ']

    Answer: No restrictions were detected.


    Allergies: Soybeans, Eggs

    Food: ['Ingredients: Enriched Corn', 'Meal (Corn Meal,', 'Ferrous Sulfate_', 'Niacin,  Thiamin', 'Mononitrate', 'Riboflavin; Folic Acid) , Vegetable Oil (Corn, Canola;', 'and/or Sunflower Oil) , Cheese Seasoning (Whey,', 'Cheddar Cheese [Milk; Cheese Cultures, Salt,', 'Enzymes]', 'Canola Oil,', 'Maltodextrin [Made from', 'Corn],', 'Natural and Artificial Flavors,', 'Salt; Whey', 'Protein  Concentrate ,', 'Monosodium', 'Glutamate', '1', 'Lactic Acid, Citric Acid, Artificial Color [Yellow 6]) ,', 'and Salt ']

    Answer: No restrictions were detected.


    Allergies: Eggs

    Food: ['INGREDIENTS:', 'Enriched', 'unbleached', 'flour', '(wheat   flour,  malted', 'flour;', "ascorbic acid [dough conditioner] ' niacin;", 'reduced', 'mononitrate ,', 'riboflavin,   folic , acid],   sugar, , degermed', 'vellow cornmeal, salt, leavening (baking', 'soda,', 'sodium', 'acid', 'pyrophosphate];', 'soybean oil, [', "'powder;, natural flavor;", 'CONTAINS; Wheat', "'contain milk; eggs, soy and tree nuts.", 'barlev', 'thiamin', 'iron,', 'honey "', 'May ']

    Answer: This contains Eggs, which was listed in your allergies.


    Allergies: Milk

    Food: ['INGREDIENTS:', 'Enriched', 'unbleached', 'flour', '(wheat   flour,  malted', 'flour;', "ascorbic acid [dough conditioner] ' niacin;", 'reduced', 'mononitrate ,', 'riboflavin,   folic , acid],   sugar, , degermed', 'vellow cornmeal, salt, leavening (baking', 'soda,', 'sodium', 'acid', 'pyrophosphate];', 'soybean oil, [', "'powder;, natural flavor;", 'CONTAINS; Wheat', "'contain milk; eggs, soy and tree nuts.", 'barlev', 'thiamin', 'iron,', 'honey "', 'May ']

    Answer: This contains Milk, which was listed in your allergies.


    Allergies: Soybeans, Tree Nuts

    Food: ['INGREDIENTS:', 'Enriched', 'unbleached', 'flour', '(wheat   flour,  malted', 'flour;', "ascorbic acid [dough conditioner] ' niacin;", 'reduced', 'mononitrate ,', 'riboflavin,   folic , acid],   sugar, , degermed', 'vellow cornmeal, salt, leavening (baking', 'soda,', 'sodium', 'acid', 'pyrophosphate];', 'soybean oil, [', "'powder;, natural flavor;", 'CONTAINS; Wheat', "'contain milk; eggs, soy and tree nuts.", 'barlev', 'thiamin', 'iron,', 'honey "', 'May ']

    Answer: This contains Soy and Tree Nuts, which were listed in your allergies.


    Allergies: Soybeans

    Food: ['INGREDIENTS:', 'Enriched', 'unbleached', 'flour', '(wheat   flour,  malted', 'flour;', "ascorbic acid [dough conditioner] ' niacin;", 'reduced', 'mononitrate ,', 'riboflavin,   folic , acid],   sugar, , degermed', 'vellow cornmeal, salt, leavening (baking', 'soda,', 'sodium', 'acid', 'pyrophosphate];', 'soybean oil, [', "'powder;, natural flavor;", 'CONTAINS; Wheat', "'contain milk; eggs, soy and tree nuts.", 'barlev', 'thiamin', 'iron,', 'honey "', 'May ']

    Answer: This contains Soy, which was listed in your allergies.


    Allergies: Fish

    Food: ['INGREDIENTS:', 'Enriched', 'unbleached', 'flour', '(wheat   flour,  malted', 'flour;', "ascorbic acid [dough conditioner] ' niacin;", 'reduced', 'mononitrate ,', 'riboflavin,   folic , acid],   sugar, , degermed', 'vellow cornmeal, salt, leavening (baking', 'soda,', 'sodium', 'acid', 'pyrophosphate];', 'soybean oil, [', "'powder;, natural flavor;", 'CONTAINS; Wheat', "'contain milk; eggs, soy and tree nuts.", 'barlev', 'thiamin', 'iron,', 'honey "', 'May ']

    Answer: No restrictions were detected.


    Allergies: {allergies}

    Food: {food}

    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["allergies", "food"],
        template= temp
    )
    return llm(prompt_template.format(allergies=allergy, food=text))

st.write("""
# Welcome to Plate Guardian!
Plate Guardian is a handy application that helps meet your dietary goals and needs. Our application provides personalized recommendations including a meal prep plan to hit your desired weight goal, and delicious recipes that won't leave you starving for your target macronutrients. Plate Guardian can also scan any food ingredient list for allergens to make you secure in your food choices.
""")
st.subheader("Dietary focus")
selected_tab = st.selectbox("Enter your dietary focus:", ["Allergens", "Meal Planner", "Macronutrients"])
def render_page(tab_name):
    if tab_name == "Allergens":
        st.subheader("Allergens")
        a = st.text_input("Enter your allergies:")   
        uploaded_file = st.file_uploader("Upload food label", type=["jpg", "jpeg", "png"])         
        if st.button('Find allergens'):
            st.write('Your allergen report is generating...')
            text = read_img(uploaded_file)
            st.write("These ingredients were found:")
            st.write(text)
            st.title(allergy(a, text))
    elif tab_name == "Meal Planner":
        st.subheader("Meal Planner")
        start = st.text_input("Enter your start weight:")    
        goal =  st.text_input("Enter your end weight:")   
        time =  st.text_input("Enter your time goal (in months):")                
        if st.button('Create meal plan'):
            st.write('Creating your meal plan...')
            st.title(plan(start, goal, time))
    elif tab_name == "Macronutrients":
        st.subheader("Macronutrients")
        c = st.selectbox("Select Macronutrient", ["Carbohydrates", "Fats", "Proteins"])           
        if st.button('Create recipe'):
            st.write('Creating a recipe targeting', c, "...")
            st.title(macro(c))
render_page(selected_tab)

