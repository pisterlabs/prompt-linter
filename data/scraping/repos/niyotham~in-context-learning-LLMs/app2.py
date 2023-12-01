from enum import unique
import os
import sys
import random

import cohere
from flask import Flask, request, jsonify, render_template

sys.path.append(os.path.abspath(os.path.join('..')))
import config

api_key = config.cohere_api['api_key']

# cohere class instance
co = cohere.Client(api_key)

# create an instance of the Flask class
app = Flask(__name__)

# list if items
items = [{"item": "item1"}, {"item": "item2"}]

@app.route('/')
def index():
    """Render the index page."""
    # Main page
    return jsonify({
                "status": "success",
                "message": "Hello, world!"
             })

# This is a simple placeholder for eccomerce, to make it dynamic we need to use a dictionary for different types of items and use the examples based on the item type
descs = [
    'Company: Casper\nProduct Name: The Wave Hybrid\nWhat is it: A mattress to improve sleep quality\nWhy is it unique: It helps with back problems\nDescription: We\'ve got your back. Literally, improving the quality of your sleep is our number one priority. We recommend checking out our Wave Hybrid mattress as it is designed specifically to provide support and pain relief.\n--SEPARATOR--\n',
    'Company: Glossier\nProduct Name: The Beauty Bag\nWhat is it: A makeup bag\nWhy is it unique: It can hold all your essentials but also fit into your purse\nDescription: Give a very warm welcome to the newest member of the Glossier family - the Beauty Bag!! It\'s the ultimate home for your routine, with serious attention to detail. See the whole shebang on Glossier.\n--SEPARATOR--\n',
    'Company: Alen\nProduct Name: Air Purifier\nWhat is it: A purifier for the air\nWhy is it unique: It\'s designed to remove allergens from the air\nDescription: The Alen BreatheSmart Classic Air Purifier is a powerful, energy-efficient air purifier that removes 99.97% of airborne particles, including dust, pollen, pet dander, mold spores, and smoke. It is designed to be used in rooms up to 1,000 square feet.\n--SEPARATOR--\n'
]


@app.route('/api/generate-description', methods=['GET', 'POST'])
def description_route():
    """description route."""
    if request.method == 'GET':
        # push the item to the list
        items.append(request.get_json())
        # return the created item
        return jsonify({
            "status": "success",
            "item": request.get_json()
        })
        # return jsonify({"status": "success", "message": "Post item!"})
    elif request.method == 'POST':
        # return generated description
        # response = co.generate(
        #     model='xlarge',
        #     prompt='Company: Casper\nProduct Name: The Wave Hybrid\nWhat is it: A mattress to improve sleep quality\nWhy is it unique: It helps with back problems\nDescription: We\'ve got your back. Literally, improving the quality of your sleep is our number one priority. We recommend checking out our Wave Hybrid mattress as it is designed specifically to provide support and pain relief.\n--SEPARATOR--\nCompany: Glossier\nProduct Name: The Beauty Bag\nWhat is it: A makeup bag\nWhy is it unique: It can hold all your essentials but also fit into your purse\nDescription: Give a very warm welcome to the newest member of the Glossier family - the Beauty Bag!! It\'s the ultimate home for your routine, with serious attention to detail. See the whole shebang on Glossier.\n--SEPARATOR--\nCompany: Cohere\nProduct Name: The FastMile\nWhat is it: A running shoe\nWhy is it unique: It\'s designed for long-distance running\nDescription:',
        #     max_tokens=50,
        #     temperature=0.9,
        #     k=0,
        #     p=0.75,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     stop_sequences=["--SEPARATOR--"],
        #     return_likelihoods='NONE'
        # )
                
        comapnay = request.get_json()['company']
        product_name = request.get_json()['product_name']
        type = request.get_json()['type']
        unique_ch = request.get_json()['unique_ch']
        
        
        # construct final string from input
        final = f"Company: {comapnay}\nProduct Name: {product_name}\nWhat is it: {type}\nWhy is it unique: {unique_ch}\nDescription:"
        
        response = co.generate(
            model='xlarge',
            # based on the item type, we can use the examples from the list, but for now we will use the same example
            prompt= descs[0] + descs[1] + final,
            max_tokens=50,
            temperature=0.9,
            k=0,
            p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=["--SEPARATOR--"],
            return_likelihoods='NONE'
        )
        
        res = response.generations[0].text
        # remove --SEPARATOR-- if x contains it
        if '--SEPARATOR--' in res:
            res = res.replace('--SEPARATOR--', '')
        
        return jsonify({"status": "success", "brand_description": res})
        # return jsonify({"status": "sucess", "message": "Get Route for items!"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.run(host='0.0.0.0', debug=True, port=port)