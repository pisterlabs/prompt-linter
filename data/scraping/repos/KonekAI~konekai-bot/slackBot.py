import os
from pathlib import Path
from dotenv import load_dotenv
import uuid
import re

from slackeventsapi import SlackEventAdapter
import slack_sdk

from flask import Flask, request, Response
import openAI as ai 


#####################################################################
#setup web server
app = Flask(__name__)

# load slack tokens from the .env file
load_dotenv(dotenv_path = Path('.') / '.env')
os.environ['SLACK_BOT_TOKEN']
#setup webClient 
client = slack_sdk.WebClient(token=os.environ['SLACK_BOT_TOKEN'])
# slack_events_adapter = SlackEventAdapter(os.environ['SIGNING_SECRET'],'/slack/events', app)

#####################################################################
# Support funcitons 

BOT_ID = client.api_call("auth.test")['user_id']

def answerFormat(question, answer):
    return [{
        'type': 'section',
        'text': {
            'type':'mrkdwn',
             'text': (
                '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
                f'*Q:* {question} \n\n'
                f'*A:* {answer}'
                )
            }
    }]
    
def wikiAnswerFormat(wikiPrompt, question, answer):
    return [{
        'type': 'section',
        'text': {
            'type':'mrkdwn',
             'text': (
                '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
                f'*Q:* {question} \n\n'
                f'*A:* According to this article about *{wikiPrompt}*... {answer}'
                )
            }
    }]
#####################################################################
# authorization
@app.route("/oauthButton", methods=["GET"])
def oAuth_generateButton():
    client_id = os.environ["SLACK_BOT_CLIENT_ID"]
    state = uuid.uuid1()
    os.putenv("TEMP_STATE", f"{state}")
    return f'<a href="https://slack.com/oauth/v2/authorize?scope=channels:read,groups:read,channels:manage,chat:write&client_id={ client_id }&state={ state }"><img alt=""Add to Slack"" height="40" width="139" src="https://platform.slack-edge.com/img/add_to_slack.png" srcset="https://platform.slack-edge.com/img/add_to_slack.png 1x, https://platform.slack-edge.com/img/add_to_slack@2x.png 2x" /></a>'

   
@app.route("/oauthRedirect", methods=["POST","GET"])
def oAuth_exchange():
    receivedState = request.args['state']
    auth_code = request.args['code']
    if(os.environ["TEMP_STATE"]) == receivedState:
        # get auth tokens from Slack
        response = client.oauth_v2_access(
            client_id=client_id,
            client_secret=client_secret,
            code=auth_code
        )
    else:
        return "Invalid State"


    os.environ["SLACK_BOT_TOKEN"] = response['access_token']

    return "Auth complete!"
    

#####################################################################
# COMMANDS
@app.route('/ai-qna', methods=['POST'])
def qna():
    payload = request.form
    user_id = payload.get('user_id')
    text = payload.get('text')

    # check if the text is empty
    if text == '':
        client.chat_postMessage(channel=user_id, 
                                text= "You did not ask me a queston. You can write your question after the command, for example (/ai-qna when did TOHack first began?)" 
                                )
    else:        
        # pass text to the AI funciton
        ans = ai.toHackTest(text)[0]
        client.chat_postMessage(channel=user_id, 
                                text= ans, 
                                blocks = answerFormat(question = text, answer = ans)
                                )
    return Response(), 200


@app.route('/wiki-read', methods=['POST'])
def set_wiki_article():
    payload = request.form
    channel_id = payload.get('channel_id')
    user_id = payload.get('user_id')
    text = payload.get('text')

    if text == '':
        client.chat_postMessage(channel= user_id, 
                                text= "Sorry, you did not give me a article name. You can write the name after the command, for example (/wiki-article Java_(programming_language))" 
                                ) 
        return Response(), 200
    # check if the text is a wiki article title
    validationCheck = ai.parseWikiText(text)
    if validationCheck == 999:
        client.chat_postMessage(channel=user_id, 
                                text= "The article you gave me does not exist on Wikipedia, please enter a valid article title" 
                                )
        return Response(), 200
    
    try:
        f = open("wiki_article.txt", "w")
        f.write(text)
        f.close()
        client.chat_postMessage(channel= channel_id, 
                        text= f"{text} ask question: /wiki-qna | remove article: /wiki-forget", 
                        blocks = [{
                                'type': 'section',
                                'text': {
                                    'type':'mrkdwn',
                                    'text': (
                                        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
                                        f'I have read the wiki article *{text}*, feel free to ask me anything via */wiki-qna*! (use */wiki-forget* once you are done with this article)'                                        
                                        )
                                    }
                            }]
                        )
        
    except:
        client.chat_postMessage(channel= user_id, 
                            text= "I was not invited to that channel. Please invite me via /invite @konekai bot so I can write to the channel!"
                            )
    return Response(), 200


@app.route('/wiki-article', methods=['POST'])
def get_wiki_article():
    payload = request.form
    user_id = payload.get('user_id')

    f = open("wiki_article.txt", "r")
    wikiArticle = f.read()
    if wikiArticle=='':
        client.chat_postMessage(channel= user_id, 
                                text= f"I am not currently reading anything, please use /wiki-read to give me an article" 
                                )
    else:
        client.chat_postMessage(channel= user_id, 
                                text= f"I am currently reading an article about {wikiArticle}"
                                )
    return Response(), 200


@app.route('/wiki-forget', methods=['POST'])
def forget_wiki_article():
    payload = request.form
    user_id = payload.get('user_id')

    f = open("wiki_article.txt", "r")
    wikiArticle = f.read()
    if wikiArticle=='':
        client.chat_postMessage(channel= user_id, 
                                text= f"I am not currently reading anything, please use /wiki-read to give me an article" 
                                )
    else:
        client.chat_postMessage(channel= user_id, 
                                text= f"I will forget this article about {wikiArticle}!"
                                )
        open("wiki_article.txt","w").close()
    return Response(), 200


@app.route('/wiki-qna', methods=['POST'])
def wiki_qna():
    payload = request.form
    channel_id = payload.get('channel_id')
    user_id = payload.get('user_id')
    text = payload.get('text')
    
    if text == '':
        client.chat_postMessage(channel=user_id, 
                                text= "You did not ask me a queston. You can write your question after the command, for example (/wiki-qna who founded Java?)" 
                                ) 
        return Response(), 200

    f = open("wiki_article.txt", "r")
    wikiArticle = f.read()
    if wikiArticle == '':
        client.chat_postMessage(channel=user_id, 
                                text= "You have not given me an article yet, you can do so via the following command (/wiki-read Java_(programming_language))" 
                                ) 
        return Response(), 200
        
    # pass text to the AI funciton
    ans = ai.getOpenAIAnswer(question = text, wikiPrompt=wikiArticle)[0]
    client.chat_postMessage(channel=channel_id, 
                            text= ans, 
                            blocks = wikiAnswerFormat(wikiPrompt=wikiArticle, question = text, answer = ans)
                            )

    return Response(), 200


# Hidden functionailties ;)
def gameFormat(game_name, link):
    return [{
        'type': 'section',
        'text': {
            'type':'mrkdwn',
             'text': (
                 '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
                'Shhh, this is a hidden feature ;) \n'
                f'*Game Suggestion: {game_name}* \n\n'
                f'{game_name} seems to be a fun social game that you can play online. Here, I suggest trying the game out on this site: {link}'
                )
            }
    }]
import random
@app.route('/game-recommendation', methods=['POST'])
def games():
    payload = request.form
    channel_id = payload.get('channel_id')
    user_id = payload.get('user_id')

    games = {'coup': 'https://coup.thebrown.net/ ',
         'Spyfall' : 'https://spyfall.adrianocola.com/', 
         'CodeName': 'https://www.horsepaste.com/', 
         'Gather town': 'https://gather.town/app',
         'Skribbl.io': 'https://skribbl.io/' 
    }
    
    game_name, link = random.choice(list(games.items())) 
    try:
        client.chat_postMessage(channel=channel_id, 
                                    text= f"Game Suggestion: {game_name}", 
                                    blocks = gameFormat(game_name = game_name, link = link)
                                    )
    except:
        client.chat_postMessage(channel=user_id, 
                            text= f"Game Suggestion: {game_name}", 
                            blocks = gameFormat(game_name = game_name, link = link)
                            )

        
    return Response(), 200



