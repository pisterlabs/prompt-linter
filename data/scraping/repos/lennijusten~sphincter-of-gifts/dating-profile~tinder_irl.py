import openai
import random
import requests
import shutil
import os
import json

# Initialize the OpenAI API client
openai.api_key = os.environ["OPENAI_API_KEY"]

# Port for the arduino
port = "/dev/cu.usbserial-130"

# counter for conversation
count = 0

# when to end conversation
CHAT_TOLERANCE = 3 #random.randint(3, 4)

# conversation history
conversation_history = ""

# alien
alien = {}

# outcome of the encounter, sets to true when the last message is a rejection (user does not receive gift)
rejected = False

def reset_count():
    global count
    global conversation_history
    global rejected
    count = 0
    conversation_history = ""
    rejected = False

def get_count():
    global count
    return count

def parse_json():
    global alien
    with open('./alien-profiles.json', 'r') as file:
        data = json.load(file)
    profiles = data["alienProfiles"]

    selected_profile = random.choice(profiles)

    name = selected_profile['name']
    bio = selected_profile['bio']
    interests = selected_profile['interests']
    planet_name = selected_profile['planetName']
    age = selected_profile['age']
    img = selected_profile['img']

    alien = {"name": name,
            "bio": bio,
            "interests": interests,
            "planet": planet_name,
            "age": age,
            "img": "img/" + img}
    return alien

def alien_traits(bio):
    input_text = bio
    response = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[{"role": "system", "content": 'You are an AI assistant that parses dating app bios, \
                  and can distill them down to three words separated by commas that describe the person whose bio that is.'},
                            {"role": "user", "content": input_text}
                  ])

    result = response["choices"][0]["message"]["content"]
    values_list = [value.strip() for value in result.split(',')]
    return values_list

def respond_to_user(prompt, role_prompt):
    global count
    global CHAT_TOLERANCE
    global conversation_history
    global rejected
    if (count <= CHAT_TOLERANCE):
        input_text = conversation_history + "\nUser: " + prompt + "\nAlien:"
        response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=[{"role": "system", "content": role_prompt},
                                {"role": "user", "content": input_text}
                      ])

        result = response["choices"][0]["message"]["content"]

        conversation_history = conversation_history + "\nUser: " + prompt + "\nAlien: " + result
        count += 1
        if (count >= CHAT_TOLERANCE):
            end = True
            #PRINT THE TICKET HERE
        else:
            end = False
        return result, end#, rejected

def get_pickup_line():
    user_input = input("What's your best pickup line for this alien?: ")
    return user_input
    # return get_user_guess_from_frontend()

def get_message():
    user_input = input("You: ")
    return user_input

def pass_pickup_line():
    return random.choice([True, True, False])
    # if random.random() < 0.5:
    #     return True
    # else:
    #     return False
    
def make_limerick(conversation_history):
    # get chatgpt to summarize the chat and date idea in one sentence
    return openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are an fun AI assistant that specializes in flirty pickup lines. You love puns and wordplay."}, 
                              {"role": "user", "content": "Write one short flirty seductive pickup line from an alien to a human."}
                              ],
                )["choices"][0]["message"]["content"]

def print_ticket(prompt: str = "SET THE PROMPT"):
    from arduino import Arduino, gen_ticket_id

    # if "|" in prompt:
    #     print("WARNING: The prompt contains a pipe character, which is used to separate the prompt from the ticket ID. Removing it.")
    #     prompt = prompt.replace("|", "")

    arduino = Arduino(port)
    arduino.wait_until_ready()
    print(arduino.recv().decode('utf-8'))
    arduino.write("Thanks for the fondle, here's a drink.")
    # arduino.write(f"{prompt}|{gen_ticket_id()}")

def generate_role_prompt(alien):
    global rejected
    print(CHAT_TOLERANCE)
    print(alien)
    role_prompt = """You are an alien looking for love, but you have high standards!
    You are chatting with the user on a dating app.
    You are not very familiar with Earth culture, but appreciate it.
    Assume an eccentric alien personality based on your traits who responds with high temperature hot takes, sometimes the sexy kind.""" + \
    " Your traits are " + ' and '.join(alien_traits(alien["bio"])) + ". " + \
    " You like " + ' and '.join(alien["interests"]) + ". "

    global count
    if (count <= CHAT_TOLERANCE):
        if count != CHAT_TOLERANCE-1:
            role_prompt += "You are interested in the User as a potential mate because you find them sexy. You are curious what the user wrote. Flirt with them seductively in a funny or unusual way. "
        else:
            if (pass_pickup_line()): # 50-50 accept or reject on the last message
                role_prompt += "You want to mate with the User because you find them sexy. You really like what the user wrote. Ask them on a date for tomorrow night seductively. "
            else: 
                rejected = True
                role_prompt += "You don't like the user and don't want to mate with them. You find them ugly and think they have a bad personality. Reject them rudely and say goodbye, but in character and acknowledging the conversation history. "

    role_prompt += """Your messages must be under 300 characters.
    Reference the user's message in your reply.
    Reference your traits and your likes if they are relevant.
    Only give the words that you would say to the user.
    Do not write your role. """

    return role_prompt

def main_game_loop():
    print("entering main game loop")
    alien = parse_json()
    print(alien["name"] + " from " + alien["planet"])
    print(alien["age"])

    # print(alien_traits(alien["bio"]))

    global conversation_history
    while (True):
        role_prompt = generate_role_prompt(alien)
        print(role_prompt)
        result = respond_to_user(get_message(), role_prompt)
        print(result[0])
        if result[-1]: # end
            break
        # print(conversation_history)

    # get_pickup_line()

def display_message_to_user(message):
    # Placeholder function to display a message on the GUI
    pass

if __name__ == "__main__":
    print_ticket("Hello world!!!!")
    quit()
    main_game_loop()
