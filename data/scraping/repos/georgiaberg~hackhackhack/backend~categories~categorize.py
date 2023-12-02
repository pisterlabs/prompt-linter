# contributors: georgiaberg, momalekabid
from flask import request, jsonify
from flask_restful import Resource
import cohere
from collections import defaultdict
import datetime
from cohere.responses.classify import Example


sentiments = ["Positive", "Neutral", "Negative"]
types = ["Experience", "Goal", "Relationship", "To-do"]

semexamples = [
    Example("Alex is always so supportive and kind", "Positive"),
    Example("Sarah has an incredible sense of humor", "Positive"),
    Example("David is a true leader and inspires others", "Positive"),
    Example("Emma is a great listener and offers valuable advice", "Positive"),
    Example("Olivia's positivity is infectious", "Positive"),
    Example("Liam is incredibly talented and hardworking", "Positive"),
    Example("Ava's generosity knows no bounds", "Positive"),
    Example("Jackson's determination is truly admirable", "Positive"),
    Example("Sophia has a heart of stone", "Negative"),
    Example("Mason's cruelty knows no bounds", "Negative"),
    Example(
        "I avoid seeking professional help for my mental health concerns", "Negative"
    ),
    Example(
        "I'm grateful for the breathtaking beauty of a colorful sunrise", "Positive"
    ),
    Example("I appreciate the soothing sound of rain on a quiet afternoon", "Positive"),
    Example("I'm thankful for the vibrant colors of autumn leaves", "Positive"),
    Example(
        "I feel gratitude for the refreshing scent of pine trees in a forest",
        "Positive",
    ),
    Example("Today, I woke up at 7:00 AM.", "Neutral"),
    Example("I made a to-do list for tomorrow.", "Neutral"),
    Example(
        "I received an email from my coworker about a meeting next week.", "Neutral"
    ),
    Example("I did some online shopping and ordered a new pair of shoes.", "Neutral"),
    Example("I cleaned the kitchen and did the dishes.", "Neutral"),
    Example("I listened to a podcast while doing household chores.", "Neutral"),
    Example("I wrote a letter to my grandmother.", "Neutral"),
    Example("I watered the plants on the balcony.", "Neutral"),
    Example("I took a hot shower before bed.", "Neutral"),
    Example("I organized my closet and donated some clothes.", "Neutral"),
    Example("I practiced a few new chords on my guitar.", "Positive"),
    Example("I did some meditation to relax before sleep.", "Positive"),
    Example("I updated my budget spreadsheet for the month.", "Neutral"),
    Example(
        "I was running late all day, and then ended up missing a lecture.", "Negative"
    ),
    Example("The weather really sucked today.", "Negative"),
    Example("I went to bed around 11:00 PM.", "Neutral"),
    Example("I cried all day.", "Negative"),
    Example("I had pains and cramps", "Negative"),
    Example("My aunt died. I'm in mourning. I haven't been able to sleep.", "Negative"),
    Example("I'm overwhelmed and not sure how to cope.", "Negative"),
    Example("Everything is working out really well", "Positive"),
    Example("I need to start coming to class early.", "Neutral"),
    Example("I need to stop smoking.", "Neutral"),
    Example("I'm so excited for the next steps in my life.", "Positive"),
]

examples = [
    Example("I want to quit smoking and lead a tobacco-free life.", "Goal"),
    Example("I aim to land a job at a prestigious company like Citadel.", "Goal"),
    Example(
        "I wish to learn to dance and become proficient in a specific style, such as salsa or hip-hop.",
        "Goal",
    ),
    Example("Do they truly understand me?", "Relationship"),
    Example("Are we growing together or apart?", "Relationship"),
    Example("What are their long-term goals for our relationship?", "Relationship"),
    Example("I dream of traveling to at least three different continents.", "Goal"),
    Example("I'm saving up for my dream vacation to a specific destination.", "Goal"),
    Example(
        "I hope to find a life partner and build a loving, long-term relationship.",
        "Goal",
    ),
    Example(
        "The gentle rustling of leaves in the breeze on an autumn day.", "Experience"
    ),
    Example("The scent of freshly cut grass on a warm summer afternoon.", "Experience"),
    Example("The feeling of a raindrop landing softly on your skin.", "Experience"),
    Example("Complete the report for the upcoming meeting.", "To-do"),
    Example("Buy groceries for the week.", "To-do"),
    Example("Schedule a dentist appointment.", "To-do"),
    Example("Pay the monthly rent or mortgage.", "To-do"),
    Example(
        "I wish I could improve my physical fitness and lead a healthier lifestyle.",
        "Goal",
    ),
    Example("I wish I had more patience in dealing with difficult situations.", "Goal"),
    Example(
        "I wish I could spend more time in nature and disconnect from technology.",
        "Goal",
    ),
    Example("I wish I had the skills to start my own business.", "Goal"),
    Example("Alex is my rock, and I trust him completely.", "Relationship"),
    Example(
        "Jennifer's wisdom has guided me through many tough times.", "Relationship"
    ),
    Example("Lucas and I share a connection that transcends words.", "Relationship"),
    Example("I'm constantly inspired by Sarah's creative talents.", "Relationship"),
]


# categories to leverage: experience, to-do, relationship, goal
def catty(note):
    string = note["content"]
    delimiters = ["?", ".", "!"]
    result = [string]

    for delimiter in delimiters:
        result = [item for sub in result for item in sub.split(delimiter) if item != ""]

    responseIntention = co.classify(model="large", inputs=result, examples=examples)
    responseEmotion = co.classify(model="large", inputs=result, examples=semexamples)

    emotionSum = 0
    counts = {"Goal": 0, "Experience": 0, "Relationship": 0, "To-do": 0}

    for i, (intention, emotion) in enumerate(zip(responseIntention, responseEmotion)):
        if emotion.prediction == "Positive":
            emotionSum += 1
        elif emotion.prediction == "Negative":
            emotionSum -= 1

        if intention.prediction in counts:
            counts[intention.prediction] += 1

    note["sentiment"] = (
        "Positive" if emotionSum > 1 else "Negative" if emotionSum < 0 else "Neutral"
    )

    # Find the top two types
    top_two_types = sorted(counts, key=counts.get, reverse=True)[:2]
    note["types"] = top_two_types

    return note
