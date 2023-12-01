import cohere
from examples import exampleList, promptPreface, promptPrefaceV2
from credentials import MODELV1, MODELV2

import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv("COHERE_KEY"))

def classify_text(text):
    classifications = co.classify(
        model="embed-english-v2.0",
        inputs=[text],
        examples=exampleList
    )
    return classifications.classifications[0].predictions[0]



def generate_response(message):
    text = format_promptV2(message)
    co = cohere.Client(os.getenv("COHERE_TRIAL_KEY"))
    # MODEL_ID = MODELV1
    MODEL_ID = MODELV2

    response = co.generate(
        model=MODEL_ID,
        prompt=text,
        max_tokens=300,
        temperature=1.5,
        k=3,
    )

    print(response.generations[0].text)
    return response.generations[0].text

def format_prompt(message):
    return promptPreface + "Girlfriend: " + message + "\n" + "User:"

def format_promptV2(message):
    return promptPrefaceV2 + "Friend: " + message + "\n" + "User:"

def summarize_text(message):
    co = cohere.Client(os.getenv("COHERE_TRIAL_KEY"))
    textIn = f"Summarize this text from Girlfriend:\nGirlfriend: Heyyy!🥳Just got back from an absolutely epic night out with Samantha, Angela, and Luna, and I can\'t wait to spill all the deets to you! 😁\n\nSo, Samantha was the star of the night, taking over the dance floor with her killer moves. Seriously, babe, she had everyone\'s jaws dropping! Luna recently broke up with her boyfriend. omg I\'m so happy for her because he was so toxic and srsly just not good enough for her.\n\nWish you were here to experience all the craziness with us. I miss you so much! 😘What are you up to tonight? 💕\nTLDR: Samantha was a killer dancer. Luna broke up with her boyfriend. Girlfriend is happy for her. Girlfriend misses you and asks what you are doing.\n\n--\n\nSummarize this text from Girlfriend:\nGirlfriend: Hey there, I just got back from school, and I\'m seriously at my wits\' end. 😔 This workload is driving me crazy, and I can\'t help but vent to you. Ugh, I\'ve got so many projects and assignments piling up, it\'s like they\'re conspiring against me!\n\nI feel like I\'m drowning in a sea of deadlines and stress, and it\'s making me so sad and frustrated. I just needed to talk to someone who gets it, and that\'s you. 😞\n\nI can\'t wait for this day to be over and for your comforting presence to make it all better.\nTLDR: Girlfriend is stressed and sad due to her many projects and schoolwork.\n\n--\n\nSummarize this text from Girlfriend:\nGirlfriend: Hey, I know you\'re probably busy, but I\'m feeling kinda down.😔 It hurts when you don\'t reply, and I can\'t help but feel a bit neglected. I really miss our chats, and your messages brighten up my day.I\'m soooooo sad, I miss you so much. Please reply to me more often it\'s really getting out of hand. Especially when you go out with your friends and I\'m not able to contact you for a while. I feel so lonely.\nTLDR: Girlfriend is lonely because she misses talking with you. She wants you to try to reply more often.\n\n--\n\nSummarize this text from Girlfriend:\nGirlfriend: Hey! Work today was pretty usual, but something funny happened afterward. So, me and Jess were grabbing coffee, and guess who we ran into - Mike! Ugh, you remember him, right? That guy from high school who always got on our nerves? Anyway, he started bragging about his job, his car, and honestly, it was hard not to roll our eyes. Jess and I couldn't help but exchange knowing glances. We were like, some things never change... But enough about that guy. How was your day, love? \nTLDR:Girlfriend and Jess ran into Mike. They don't like him because Mike brags a lot.\n\n--\n\nSummarize this text from Girlfriend:\nGirlfriend:{message}\nTLDR:"
    # pinrt
    response = co.summarize( 
        text= textIn,
        length='short',
        format='bullets',
        model='command',
        extractiveness="medium",
        additional_command='focus on names mentioned and events that occured. No more than 3 bullets',
        temperature=0.5,
    ) 
    return response.summary

''' ==========================FUNCTION CALLS (for testing)=========================== '''
# generate_response("would you love me if i were a worm?")
