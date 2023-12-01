import os
from dotenv import load_dotenv
import openai
from flask import Flask, render_template, request


# added in the static_folder to allow for the css to be used
app = Flask(__name__, static_folder='templates/assets')

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/recipeGeneration', methods=['POST'])
def recipeGeneration():
    protein = request.form.get('protein')
    carb = request.form.get('carb')
    fat = request.form.get('fat')
    calorie = request.form.get('calorie')
    otherInfo = request.form.get('otherInfo')
    
    load_dotenv()

    API_KEY = os.getenv("OPENAI_API_KEY")
    ORG_KEY = os.getenv("ORG_KEY")

    openai.organization = ORG_KEY
    openai.api_key = API_KEY
    print("Running AI")
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Give me a recipe that has around " + protein + " grams of protein, " + carb + " grams of carbohydrate, " +
                fat + " grams of fat and has " + calorie + " calories. Along with that include this information: " + otherInfo + " For the Response please first calculate and list the macronutrients and calorie count. After that make 2 new lines then  list of ingredents followed by 3  new lines and then  the cooking instructions."}
        ]
    )
    print("Done with AI call")
    
    recipe = completion.choices[0].message.content
    print(recipe)
    return str(recipe)



@app.route('/recipeGenerationFullDay', methods=['POST'])
def recipeGenerationFullDay():
    protein = request.form.get('protein')
    carb = request.form.get('carb')
    fat = request.form.get('fat')
    calorie = request.form.get('calorie')
    otherInfo = request.form.get('otherInfo')
    
    load_dotenv()

    API_KEY = os.getenv("OPENAI_API_KEY")
    ORG_KEY = os.getenv("ORG_KEY")

    openai.organization = ORG_KEY
    openai.api_key = API_KEY

    prompt = "For the response I would like a full day of eating that satisfies the given information. In the full day I want there to be " + protein + " grams of protein, " + carb + " grams of carbs, " + fat + " grams of fat. At around a total of " + calorie + " calories for the day. Along with this incorporate the following information for the meal plan: " + otherInfo + ". In the response begin with how many meals there are, the macro nutrient count in each meal and then the keep each recipe organized. "
    print("Running AI")
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    print("Done with AI call")
    
    recipe = completion.choices[0].message.content
    print(recipe)
    return str(recipe)

if __name__ == "__main__":
    app.run()



