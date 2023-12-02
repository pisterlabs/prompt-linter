#!/usr/bin/env python
# -*- coding: utf-8 -*-

#       _                              
#      | |                             
#    __| |_ __ ___  __ _ _ __ ___  ___ 
#   / _` | '__/ _ \/ _` | '_ ` _ \/ __|
#  | (_| | | |  __/ (_| | | | | | \__ \
#   \__,_|_|  \___|\__,_|_| |_| |_|___/ .
#
# A 'Fog Creek'–inspired demo by Kenneth Reitz™

import os
from flask import Flask, request, render_template, jsonify
import cohere
from cohere.classify import Example
import random
from flask_cors import CORS

# initialize the Cohere Client with an API Key
co = cohere.Client(os.environ.get('API_KEY'))

# Support for gomix's 'front-end' and 'back-end' UI.
app = Flask(__name__)
cors = CORS(app, resources={r"/": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})

# Set the app secret key from the secret environment variables.
app.secret = os.environ.get('SECRET')

# Dream database. Store dreams in memory for now. 
DREAMS = ['Python. Python, everywhere.']
@app.route('/notify', methods=['GET'])
def send_notification():
    return 
                               
                
# CONVERSATION DATABASE
questionnaire = {
  "I normally have enough energy to get through the day": False, 
  "I generally get 8 hours of sleep a day": True,
  "I feel like I get enough exercise": False,
  "I'm happy with the connections I have with my coworkers": True,
  "I feel close with my friends and family": True,
  "I feel fulfilled by the work I do": False,
  "I have hobbies outside of work": True,
  "I feel like I have a balanced diet": False,
  "I enjoy cooking my meals": False,
  "I have time to cook my meals": False
}

def msgSentiment(userMsg):
  userMsgs = []
  userMsgs.append(userMsg)

  sentimentExamples=[
    Example("I've been doing great!", "positive"), 
    Example("I'm feeling better", "positive"), 
    Example("Pretty good! I spent yesterday night with my friends and we had a great time.", "positive"), 
    Example("I've been using the strategies you've given me and they've been working well", "positive"), 
    Example("I've been eating healthier recently", "positive"),
    Example("I've been eating healthier recently but I hate doing it", "negative"), 
    Example("I've been sick the past couple of days.", "negative"), 
    Example("I'm very tired", "negative"), 
    Example("Not very good. I've been very stressed with the upcoming project deadline", "negative"), 
    Example("I don't think the strategies you recommended work for me.", "negative"), 
    Example("I didn't eat anything for lunch", "negative"), 
    Example("I could be better...", "negative"), 
    Example("Not too bad. Nothing super eventful happened this week though.", "neutral"), 
    Example("I'm okay", "neutral"), 
    Example("Alright.", "neutral"), 
    Example("I don't know", "neutral"), 
    Example("I ate cereal for breakfast", "neutral"), 
    Example("My favourite TV show is Community", "neutral"), 
    Example("I know how to drive", "neutral")
  ]
  
  response = co.classify(
  model='medium',
  inputs=userMsgs,
  examples=sentimentExamples,
  )
  
  sentiment = response.classifications[0].prediction

  return sentiment

  
def msgReccomendation(botQ, userMsg):
  msgClassExamples=[
    Example("I've been doing great!", "mood"), 
    Example("I'm feeling better", "mood"), 
    Example("Pretty good! I spent yesterday night with my friends and we had a great time.", "social"), 
    Example("I've been using the strategies you've given me and they've been working well", "mood"), 
    Example("I've been eating healthier recently", "diet"), 
    Example("I've been sick the past couple of days.", "health"), 
    Example("I'm very tired", "sleep"), 
    Example("Not very good. I've been very stressed with the upcoming project deadline", "mood"), 
    Example("I don't think the strategies you recommended work for me.", "mood"), 
    Example("I didn't eat anything for lunch", "diet"), 
    Example("I could be better...", "mood"), 
    Example("Not too bad. Nothing super eventful happened this week though.", "mood"), 
    Example("I'm okay", "mood"), 
    Example("Alright.", "mood"),
    
    Example("Have you tried cooking with friends or family?", "diet"), 
    Example("Have you tried reaching out to a friend this week?", "social"), 
    Example("Have you had the time to go grocery shopping recently?", "diet"), 
    Example("Were you able to make any time to exercise yesterday?", "exercise"), 
    Example("I've started running in the mornings!", "exercise"), 
    Example("I'm feeling really well rested", "sleep"), 
  ]
  pass

def constructBotQuestion():
  questionBasis = []
  
  if not questionnaire.get("I normally have enough energy to get through the day"):
    questionBasis.append("I always feel super tired.")
  if not questionnaire.get("I generally get 8 hours of sleep a day"):
    questionBasis.append("I don't get enough sleep.")
  if not questionnaire.get("I feel like I get enough exercise"):
    questionBasis.append("I don't feel like I get enough exercise.")
  if not questionnaire.get("I'm happy with the connections I have with my coworkers"):
    questionBasis.append("I don't have very good relationships with my coworkers.")
  if not questionnaire.get("I feel close with my friends and family"):
    questionBasis.append("I feel disconnected from my friends.")
    questionBasis.append("I feel disconnected from my family.")
  if not questionnaire.get("I feel fulfilled by the work I do"):
    questionBasis.append("I don't feel like the work I do is meaningful.")
  if not questionnaire.get("I have hobbies outside of work"):
    questionBasis.append("I don't have any hobbies outside of work.")
  if not questionnaire.get("I feel like I have a balanced diet"):
    questionBasis.append("I don't think I have enough variety in my diet.")
    questionBasis.append("I don't think the meals I eat are very nutritious.")
  if not questionnaire.get("I enjoy cooking my meals"):
    questionBasis.append("I hate cooking.")
  if not questionnaire.get("I have time to cook my meals"):
    questionBasis.append("I don't have enough time to cook my own meals.")
  
  prompt = f"""
  User: I always feel super tired.
  Bot: Have you had the chance to go outside this week?
  --
  User: I always feel super tired.
  Bot: Did you get enough sleep last night?
  --
  User: I hate cooking!
  Bot: Have you tried cooking with friends or family?
  --
  User: I never have enough time to cook.
  Bot: Did you make time to take care of yourself yesterday?
  --
  User: I never have the time to exercise.
  Bot: Were you able to make any time to exercise yesterday?
  --
  User: I never have the time to exercise.
  Bot: Are there any exercises that you enjoy doing?
  --
  User: I've been feeling really disconnected from my friends.
  Bot: Have you tried reaching out to a friend this week?
  --
  User: I never have enough time to cook.
  Bot: Have you had the time to go grocery shopping recently?
  --
  User: I don't have any hobbies outside of work!
  Bot: Have you tried doing something new with a friend?
  --"""
  
  userSegment = questionBasis[random.randint(0, (len(questionBasis) - 1))]
  prompt = prompt + "\nUser: " + userSegment
  prompt = prompt + "\nBot:"
  
  question = co.generate(  
    model='large',  
    prompt = prompt,  
    max_tokens=40,  
    temperature=0.8,  
    stop_sequences=["--"],
    p = 0.6,
    presence_penalty = 0.4
  )
  
  botQuestion = question.generations[0].text
  
  return botQuestion

def constructBotResponse(botQ, userMsg):
  userSentiment = msgSentiment(userMsg)
    
  prompt = f"""
  Bot: How are you feeling today?
  User: positive
  Bot: That's good to hear!
  --
  Bot: How are you feeling today?
  User: positive
  Bot: That's awesome!
  --
  Bot: How are you feeling today?
  User: negative
  Bot: I'm sorry to hear that, hopefully our chat will help you feel a little better!
  --
  Bot: How are you feeling today?
  User: negative
  Bot: It seems like today hasn't gotten off to a good start. I hope our chat today can help with that!
  --
  Bot: Do you think you've been meeting your goals today?
  User: positive
  Bot: I'm glad you've been doing so well!
  --
  Bot: Do you think you've been meeting your goals today?
  User: negative
  Bot: That's okay! Maybe some of the strategies from today's chat will help you accomplish what you'd like to today.
  --
  Bot: Were you able to cook something yesterday?
  User: negative
  Bot: That's okay! We all have days where cooking is difficult.
  --
  Bot: What is your favourite recipe?
  User: neutral
  Bot: That sounds good!
  --
  Bot: What is your favourite recipe?
  User: neutral
  Bot: You have great taste.
  --
  Bot: Have you been spending time with your family?
  User: neutral
  Bot: Interesting!
  --
  Bot: Do you think you've been meeting your goals today?
  User: negative
  Bot: Oh no! Hopefully our chat will help you feel more confident in your goal setting.
  --"""
  prompt = prompt + "\nBot: " + botQ
  prompt = prompt + "\nUser: " + userSentiment
  prompt = prompt + "\nBot:"
  
  response = co.generate(  
    model='large',  
    prompt = prompt,  
    max_tokens=30,  
    temperature=0.8,  
    stop_sequences=["--"],
    p = 0.55,
    frequency_penalty = 0.4,
    logit_bias = {'64': -10}
  )

  botResponseFull = response.generations[0].text
  
  # condense bot response into 2 sentences
  botSentences = []
  botResponseFull.replace('!', '.').replace('?', '.')
  botSentences = botResponseFull.split('.')
  
  botResponse = botSentences[0] + '.'
  
  return botResponse

@app.route('/', methods=['POST'])
def dreams():
  content = request.json
  botQ = content['botQ']
  userMsg = content['userMsg']
  botQuestion = constructBotQuestion()
  botResponse = constructBotResponse(botQ, userMsg)
  
  botReturn = botResponse + botQuestion
  
  return botReturn
# POST
  # Input: Analyze the conversation's whole sentiment (JUST SEND THE USER MESSAGES)
  # Output: Colours and how much of each colour for the sentiment
"""

"""
# 

if __name__ == '__main__':
    app.run()
