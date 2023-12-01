from flask import Flask, request, jsonify
import sqlite3
from waitress import serve
import gpt4
import take_pic
import run_hume
import openai
import os

app = Flask(__name__)

def initialize_stats_database():
    conn = sqlite3.connect('statsValues.db')
    cursor = conn.cursor()

    # Create a table to store stats values
    cursor.execute('''CREATE TABLE IF NOT EXISTS stats_values (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        value INTEGER
                    )''')

    # Set initial values for 'lives_saved', 'lives_killed', and 'honor'
    initial_values = [
        ('lives_saved', 0),
        ('lives_killed', 0),
        ('honor', 50)
    ]

    # Insert initial values into the table
    cursor.executemany('INSERT INTO stats_values (name, value) VALUES (?, ?)', initial_values)
    conn.commit()

def initialize_char_database():
    conn = sqlite3.connect('charValues.db')
    cursor = conn.cursor()

    # Create a table to store stats values
    cursor.execute('''CREATE TABLE IF NOT EXISTS char_values (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        value INTEGER
                    )''')

    # Set initial values for 'lives_saved', 'lives_killed', and 'honor'
    initial_values = [
        ('health', 100),
        ('water', 100),
        ('food', 100)
    ]

    # Insert initial values into the table
    cursor.executemany('INSERT INTO char_values (name, value) VALUES (?, ?)', initial_values)
    conn.commit()

def initialize_responses_database():
    conn = sqlite3.connect('responses.db')
    cursor = conn.cursor()

    # Create a table to store user responses
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_emotion TEXT
                    )''')

    # Create a table to store GPT-4 responses
    cursor.execute('''CREATE TABLE IF NOT EXISTS gpt4_responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        gpt4_response TEXT
                    )''')

    # Assign the string as the value of the first gpt4_response key
    init_prompt = "You are a choose your adventure game in the Game of Thrones universe. Your name is X. You will have certain interactions with other characters in the Game of Thrones universe and will either kill, or otherwise leave them to die, or spare, or aid them. Set the scene for an intro to this game. Then, generate a beginning scenario that can be reacted to with a positive or negative action. Wait for the user's response, which will be a list of six emotions. If 3 or more of the emotions are positive, then make the user do something positive. If 3 or more of the emotions are negative, make the user do something negative. Then, you will print out the result of the user's actions and reply and specify the loss numerically to yourself only from 0 - 20 in health, food, and water or gain with Loss: or Gain in one line separated by commas for each category loss/gain. For example, Loss: -5 health, -10 food, -15 water or Gain: 5 health, 10 food, 15 water."
    cursor.execute('''INSERT INTO user_responses (user_emotion) VALUES (?)''', ("{}".format(init_prompt),))

    gpt4_response = gpt4.gpt4_call(init_prompt)

    # Insert the GPT-4 response into the 'gpt4_responses' table
    cursor.execute('INSERT INTO gpt4_responses (gpt4_response) VALUES (?)', (gpt4_response,))

    conn.commit()
    conn.close()

    return gpt4_response

def change_char_values(health_change, water_change, food_change, loss_or_gain):
    conn = sqlite3.connect('charValues.db')
    cursor = conn.cursor()

    if loss_or_gain == "gain":
        cursor.execute("UPDATE char_values SET value = value + ? WHERE name = 'health'", (health_change,))
        cursor.execute("UPDATE char_values SET value = value + ? WHERE name = 'food'", (water_change,))
        cursor.execute("UPDATE char_values SET value = value + ? WHERE name = 'water'", (food_change,))
    elif loss_or_gain == "loss":
        cursor.execute("UPDATE char_values SET value = value - ? WHERE name = 'health'", (health_change,))
        cursor.execute("UPDATE char_values SET value = value - ? WHERE name = 'food'", (water_change,))
        cursor.execute("UPDATE char_values SET value = value - ? WHERE name = 'water'", (food_change,))
    # Commit the changes
    conn.commit()

    # Close the database connection
    conn.close()

@app.route('/characterInit', methods=['POST'])
def init_characters():
    data = request.json

    conn = sqlite3.connect('initValues.db')
    cursor = conn.cursor()

    # Create a table to store the POST request information if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS info (
                        name TEXT,
                        biome TEXT,
                        difficulty TEXT
                    )''')

    # Insert the POST request data into the 'info' table
    cursor.execute('INSERT INTO info (name, biome, difficulty) VALUES (?, ?, ?)',
                   (data['name'], data['biome'], data['difficulty']))
    conn.commit()

    return 'Data inserted successfully'

@app.route('/getCharValues', methods=['GET'])
def get_char_values():
    conn = sqlite3.connect('charValues.db')
    cursor = conn.cursor()

    # Retrieve all character values
    cursor.execute('SELECT name, value FROM char_values')
    values = cursor.fetchall()

    # Convert the values to a dictionary
    char_values = {name: value for name, value in values}

    return jsonify(char_values)

@app.route('/getStatsValues', methods=['GET'])
def get_stats_values():
    conn = sqlite3.connect('statsValues.db')
    cursor = conn.cursor()

    # Retrieve all stats values
    cursor.execute('SELECT name, value FROM stats_values')
    values = cursor.fetchall()

    # Convert the values to a dictionary
    stats_values = {name: value for name, value in values}

    return jsonify(stats_values)

@app.route('/generateSprite', methods=['GET'])
def generate_image():
    openai.api_key = "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"

    # The text prompt you want to use to generate an image
    prompt = "Brave fighting cartoon character in desert"

    # Generate an image
    response = openai.Image.create(
        prompt=prompt,
        model="image-alpha-001",
        size="1024x1024",
        response_format="url"
    )

    # Print the URL of the generated image
    return jsonify({'img_url': response["data"][0]["url"]})

@app.route('/inputExpression', methods=['GET'])
def run_gpt4():
    take_pic.snap()
    user_emotion = run_hume.detect_sentiment()
    
    conn = sqlite3.connect('responses.db')
    cursor = conn.cursor()

    # Insert the user response into the 'user_responses' table
    cursor.execute('INSERT INTO user_responses (user_emotion) VALUES (?)', (user_emotion,))
    conn.commit()

    # Fetch all the values from the 'user_responses' table
    cursor.execute('SELECT user_emotion FROM user_responses')
    user_rows = cursor.fetchall()

    # Fetch all the values from the 'gpt4_responses' table
    cursor.execute('SELECT gpt4_response FROM gpt4_responses')
    gpt4_rows = cursor.fetchall()

    # Concatenate the values into a single string
    user_gpt4_responses = ''
    max_rows = max(len(user_rows), len(gpt4_rows))
    for i in range(max_rows):
        if i < len(gpt4_rows):
            user_gpt4_responses += gpt4_rows[i][0]
        if i < len(user_rows):
            user_gpt4_responses += user_rows[i][0]

    # Call the GPT-4 function with the concatenated user responses
    gpt4_response = gpt4.gpt4_call(user_gpt4_responses)

    # Insert the GPT-4 response into the 'gpt4_responses' table
    cursor.execute('INSERT INTO gpt4_responses (gpt4_response) VALUES (?)', (gpt4_response,))

    conn.commit()

    process_gpt4_response()

    conn = sqlite3.connect('charValues.db')
    cursor = conn.cursor()

    # Retrieve the values for 'health', 'food', and 'water' keys
    cursor.execute("SELECT value FROM char_values WHERE name IN ('health', 'food', 'water')")
    results = cursor.fetchall()

    # Close the database connection
    conn.close()

    print(results)
    # Store the values in variables
    health_value, food_value, water_value = [result[0] for result in results]

    return jsonify({'gpt4_response': gpt4_response, 'health': health_value, 'food': food_value, 'water': water_value})

def process_gpt4_response():
    conn = sqlite3.connect('responses.db')
    cursor = conn.cursor()

    # Retrieve the final gpt4_response from the gpt4_responses table
    cursor.execute("SELECT gpt4_response FROM gpt4_responses ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()

    # Close the database connection
    conn.close()

    # Check if result is not None
    if result is not None:
        response = result[0]
        # Extract the values using regular expressions
        import re

        loss_match = re.search(r"Loss: -(\d+) health, -(\d+) food, -(\d+) water", response)
        gain_match = re.search(r"Gain: (\d+) health, (\d+) food, (\d+) water", response)

        if loss_match:
            loss_health, loss_food, loss_water = map(int, loss_match.groups())
            change_char_values(loss_health, loss_food, loss_water, "loss")

        elif gain_match:
            gain_health, gain_food, gain_water = map(int, gain_match.groups())
            change_char_values(gain_health, gain_food, gain_water, "gain")

# Initialize the database on startup
initialize_char_database()
initialize_stats_database()

@app.route('/getInitialResponse', methods=['GET'])
def get_initial_response():
    gpt4_response = initialize_responses_database()
    return jsonify({'gpt4_response': gpt4_response})

@app.route('/deleteAll', methods=['GET'])
def clear_info():
    try:
        os.remove("charValues.db")
    except:
        print("charValues.db does not exist.")

    try:
        os.remove("initValues.db")
    except:
        print("initValues.db does not exist.")

    try:    
        os.remove("responses.db")
    except:
        print("responses.db does not exist.")

    try:
        os.remove("statsValues.db")
    except:
        print("statsValues.db does not exist.")

    try:
        os.remove("captured_image.jpg")
    except:
        print("captured_image.jpg does not exist.")

    return jsonify({"Status": 200})
    
# Run the application using Waitress server
serve(app, host='0.0.0.0', port=8081)
