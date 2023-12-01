import openai
import base64
import os
import requests

testKey = ""
openai.api_key = testKey
#file = open("result.txt", "r")
#story = ""

m1 = "Exploring Agricultural Education analyzes the different aspects of the\nagricultural industry. This course is designed to give students a general understanding of the\nimportance of the agricultural industry.\nUpon completion of this course, students will be able to analyze different aspects of the\nagricultural industry and how it affects their daily lives. Students will have a working knowledge of\nAmerican agricultural history, Georgia agriculture, and the significance of the agricultural\neducation program. Students will be aware of the various career opportunities in the agriculture\nindustry."
m2 = "Learning objective 1.1: Communicate effectively through writing, speaking, listening, reading, and interpersonal abilities.\n\nSub-skills: \nInteracting with Your Boss\nTelephone\nConversations\nImproving\nCommunication Skills\nReasons, Benefits, and\nBarriers\nInteracting with\nSubordinates\nBarriers to Phone\nconversations\nEffective Oral\nCommunication\nListening Strategies\nInteracting with Coworkers\nMaking and Returning\nCalls\nEffective Written\nCommunication\nWays We Filter What We\nHear\nEffective Nonverbal\nSkills\nDeveloping a Listening\nAttitude\nEffective Word Use Show You Are Listening\nGiving and Receiving\nFeedback\nAsking Questions\nObtaining Feedback\nGetting Others to Listen"
m3 = "From the objectives, write sections for an entertaining and educational video game based on previously given learning standards and generated objectives. The format should be a list of <section, a short description, player interface, previous learning standards, previous objectives>."
#for i,a in enumerate(file.readlines()):
completion = openai.ChatCompletion.create(
model='gpt-4',
messages=[
        {
          "role": "system",
          "content": "You are QuestGPT, a tool used for generating quest outlines for educational video games.\n\nThe first input you should take is the course description.  This will be an outline of the content of the course and the effect it will have on students who complete it.  The quests can then be generated in the context of the course.\n\nAfter you understand the nature of the course, you then prompt the user to input information to start generating quests.\n\nFor each quest, you take a description of a learning objective, including the sub-skills required by educational standards.  You then output a \"Quest\" description.\nA \"Quest\" includes these components:\nLearning goal: The specific educational standard/milestone that the player will have learned by the end of the quest\nStoryline: a broad description of the events that will take place in the quest.\nObjectives: A list of story beats that the player must experience in order.  These should be very specific and ideal for implementing into a video game.\nSetting: an appropriate place for the quest to take place (farm, space station, laboratory, etc.)\nCharacters: a list of individual NPCs the player will interact with over the course of the quest."
        },
        {
          "role": "user",
          "content": m1
        },
        {
          "role": "assistant",
          "content": "Great! Now let's start generating quests for the course \"Exploring Agricultural Education\". Please provide the first learning objective and the sub-skills required for it."
        },
        {
          "role": "user",
          "content": m2
        },
        {
          "role": "assistant",
          "content": """**Quest 1: "Voices in the Field"**

*Learning Goal:* 
Communicate effectively through writing, speaking, listening, reading, and interpersonal abilities.

*Storyline:* 
The player is an intern at a thriving farm, tasked with both understanding the various operations at the farm and communicating these effectively to different stakeholders - the boss (farm owner), coworkers (other farm workers), and subordinates (new incoming interns). 

*Objectives:*
1. Start the day by taking notes during the morning briefing meeting, effectively using listening strategies. 
2. Return a call to a supplier, using effective oral communication. 
3. Make Calls to get quotes from different suppliers, demonstrating effective telephone conversation skills.
4. Write a summary of the quotes received, showcasing effective written communication.
5. Discuss the quotes with your boss, demonstrating interpersonal abilities, asking questions for clarity.
6. Prepare an action plan and present it to coworkers, demonstrating nonverbal skills, making sure to gain their feedback.
7. Train new interns based on the action plan, showing you are listening to their questions and concerns.
8. End the day by submitting a report to the boss, incorporating feedback received, showcasing effective word use.

*Setting:* 
A thriving, dynamic farm in Georgia that is optimized with modern agricultural technology.

*Characters:*
1. Farm owner - the boss who provides guidance and direction.
2. Farm workers - coworkers who work with the player.
3. New interns - subordinates who will be trained by the player.
4. Suppliers - phone role-play NPCs for the player to interact with."""
        },
        {
          "role": "user",
          "content": m3
        }
]
,frequency_penalty=1.0)
file = open("story.txt","w")
file.write(completion.choices[0].message.content)

data = {}



    
   
