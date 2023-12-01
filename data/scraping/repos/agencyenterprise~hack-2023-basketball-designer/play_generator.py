import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def query_gpt4(prompt, content, temperature=1.0, max_tokens=128, n=1):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=max_tokens,
        n = n
    )
    return response

def generate_play_descriptions(play_type, defense_type, zone_text, additional_option, play_description, num_plays):
    # This is where you'd put the code to generate a play.
    # For now, it just returns a placeholder message.
    prompt = f"You are a highly skilled, basketball coach AI trained in {play_type.lower()} play design. \
        I would like you to create a single {additional_option.lower()} where the defense is a {zone_text.lower()} {defense_type.lower()} defense. \
        These plays must adhere to the following requirements."
    content = f"Requirements: {play_description} The single play should be provided in the format 'play_name|play_description' and must be a concise, high-level description not longer than two sentences."
    responses = query_gpt4(prompt, content, n=num_plays)
    play_names = []
    play_descriptions = []
    print(responses)
    for response in responses['choices']:
        print(response['message']['content'])
        play_name, play_description = response['message']['content'].replace("\"", "").split('|')
        play_names.append(play_name)
        play_descriptions.append(play_description)
    return play_names, play_descriptions

def generate_play_by_play(play_type, play_name, play_description):
    # This is where you'd put the code to generate a play.
    # For now, it just returns a placeholder message.
    prompt = f"You are a highly skilled, basketball coach AI trained in {play_type.lower()} play design. \
        I would like you to provide a detailed, step-by-step, play-by-play description of the following play, \
        including positions on the court where each player and the ball is at the end of each step of the play."
    content = f"Play: The play is called {play_name} and has the following description: {play_description} \
        Please provide a detailed, step-by-step, play-by-play description based on this information, \
        including positions on the court where each player and the ball is at the end of each step of the play."
    response = query_gpt4(prompt, content, temperature=0.2, max_tokens=1024)
    play_by_play = response['choices'][0]['message']['content']
    return play_by_play

def generate_animation_data(play_by_play, height, width):
    # This is where you'd put the code to generate the animation data.
    # For now, it just returns a placeholder message.
    prompt = f"You are a highly skilled, basketball animator AI. \
        I would like you to provide accurate coordinate locations for each player and for the ball at each step of the play-by-play. \
        Assume each step in the play-by-play description corresponds to one time step. \
        Ensure all coordinates are within the dimensions of the horizonal full court, which is {height} pixels high and {width} pixels wide. \
        Generate the locations as if the offense is going to the right side of the court. \
        For reference, the top of the key is at {(width*2/3, height/2)}, the free throw line is at {(width*4/5, height/2)}, \
        the weak side block is at {(width*11/12, height*7/10)}, the strong side block is at {(width*11/12, height*3/10)}, \
        the weak side wing is at {(width*4/5, height*11/12)}, the strong side wing is at {(width*4/5, height*1/12)}, \
        the weak side corner is at {(width*19/20, height*19/20)}, the strong side corner is at {(width*19/20, height*1/20)}, \
        the basket is at {(width*19/20, height/2)}, and the baseline is at x={width}. \
        For each player, make sure you have an array of tuple locations (x, y) at each point in time. \
        You should then make a list of these player (and ball) arrays in the format [[(x1, y1), (x2, y2), ...], [(x1, y1), (x2, y2), ...], ...] \
        with the coordinates for the ball being the last array listed. \
        Provide just this array and nothing else. Please do not include any comments or non-Python syntax in your response."
    content = f"Play description: {play_by_play} Remember to only provide the array of player locations and nothing else in your response."
    response = query_gpt4(prompt, content, temperature=0.2, max_tokens=1024)
    locations = response['choices'][0]['message']['content']
    print(locations)
    locations = eval(locations)
    return locations