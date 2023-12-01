#SYSTEM STUFF
import os
import json
import asyncio
from copy import copy
from pathlib import Path
import openai
from human_id import generate_id
from datetime import datetime
from prisma import Json
#NOVA STUFF

from session.appHandler import app, websocket
from session.sessionHandler import novaConvo, active_cartridges, chatlog, cartdigeLookup, novaSession, current_loadout, current_config
from session.prismaHandler import prisma
from core.cartridges import copy_cartridges_from_loadout, update_cartridge_field
from chat.chat import agent_initiate_convo, construct_query
from session.tokens import update_coin_count
from tools.memory import run_summary_cartridges
from file_handling.s3 import get_signed_urls
from tools.debug import fakeResponse, eZprint

agentName = "nova"
openai.api_key = os.getenv('OPENAI_API_KEY', default=None)

 
##CARTRIDGE MANAGEMENT




async def initialise_conversation(sessionID,convoID, params = None):
    ##session setup stuff should be somewhere else
    DEBUG_KEYS = ['INITIALISE', 'INITIALISE_CONVERSATION']
    eZprint('initialising conversation', DEBUG_KEYS)
    if convoID not in novaConvo:
        novaConvo[convoID] = {}
    if params:
        # if 'fake_user' in params and params['fake_user'] == 'True':
        #     eZprint('fake user detected')
        #     novaSession[sessionID]['fake_user'] = True 
        #     novaSession[sessionID]['userName'] = "Archer"

        for key in params:
            if 'DEBUG' in key:
                os.environ[key] = params[key]
                print('setting os params ' + str(key) + ' to ' + str(params[key]))

        if 'agent_initiated' in params and params['agent_initiated'] == 'True':
            eZprint('agent initiated convo')
            novaSession[sessionID]['agent_initiated'] = True
        if 'name' in params:
            novaSession[sessionID]['user_name'] = params['name']
        if 'message' in params:
            novaConvo[convoID]['message'] = params['message']
            # print(params['message'])

        if 'model' in params:
            novaConvo[convoID]['model'] = params['model']
            if novaConvo[convoID]['model'] == 'gpt-4':
                novaConvo[convoID]['token_limit'] = 8000
            else:
                novaConvo[convoID]['token_limit'] = 4000
        else: 
            novaConvo[convoID]['token_limit'] = 4000
    if 'token_limit' not in novaConvo[convoID]:
        novaConvo[convoID]['token_limit'] = 4000
    # print('nova convo')
    # print(novaConvo[convoID])
        # print(params['model'])

    userID = novaSession[sessionID]['userID']

    await update_coin_count(userID,0)

    novaConvo[convoID]['token_limit'] = 4000
    
    # novaConvo[convoID]['agent-name'] = agentName
    # sessionID = novaConvo[convoID]['sessionID'] 
    novaSession[sessionID]['latestConvo'] = convoID
    

async def initialiseCartridges(sessionID, convoID, loadout):
    DEBUG_KEYS = ['INITIALISE', 'INITIALISE_CARTRIDGES']
    eZprint('intialising cartridges',DEBUG_KEYS)
    # print(current_loadout[sessionID])
    # if sessionID not in current_loadout:
    #     current_loadout[sessionID] = None
    # if sessionID not in current_loadout or current_loadout[sessionID] == None:
        ## get rid of this, basically accounts for 'none' loadout which I want to scrap
    if not loadout: 
        await loadCartridges(sessionID, convoID)
        novaSession[sessionID]['owner'] = True
        await websocket.send(json.dumps({'event': 'set_config', 'payload':{'config': current_config[sessionID], 'owner': novaSession[sessionID]['owner']}}))
 
    await runCartridges(sessionID, loadout)


async def loadCartridges(sessionID, convoID, loadout = None):
    DEBUG_KEYS = ['INITIALISE', 'LOAD_CARTRIDGES']
    eZprint('load cartridges called', DEBUG_KEYS)
    userID = novaSession[sessionID]['userID']

    cartridges = await prisma.cartridge.find_many(
        where = {  
        "UserID": userID,
        }
    )

    active_cartridges[convoID] = {}
    if len(cartridges) != 0:
        for cartridge in cartridges:    
            blob = json.loads(cartridge.json())
            for cartKey, cartVal in blob['blob'].items():
                if 'softDelete' not in cartVal or cartVal['softDelete'] == False:
                    active_cartridges[convoID][cartKey] = cartVal

        await websocket.send(json.dumps({'event': 'sendCartridges', 'cartridges': active_cartridges[convoID], 'convoID': convoID }))

async def runCartridges(sessionID,  convoID, loadout = None):
    DEBUG_KEYS = ['INITIALISE', 'RUN_CARTRIDGES']
    eZprint('running cartridges', DEBUG_KEYS)
        
    # if sessionID in current_config:
    #     print('current config')
    #     # print(current_config[convoID])
    #     if 'agent_initiated' in current_config[sessionID] and current_config[sessionID]['agent_initiated'] == True:
    #         await agent_initiate_convo(sessionID, convoID, loadout)

    if convoID in active_cartridges:
        for cartKey, cartVal in active_cartridges[convoID].items():
            if cartVal['type'] == 'media':
                file_to_request = cartKey
                if cartVal.get('aws_key'):
                    file_to_request = cartVal['aws_key']
                url = await get_signed_urls(file_to_request)
                await update_cartridge_field({'cartKey': cartKey, 'sessionID': sessionID, 'fields': {'url': url}}, loadout)
            # if cartVal['type'] == 'summary':
            #     if 'enabled' in cartVal and cartVal['enabled'] == True:
            #         # print('running summary cartridge on loadout ' + str(loadout))
            #         # if cartVal['state'] != 'loading':
            #         # print('running summary cartridge' + str(cartVal))
            #         # if 'running' in cartVal:
            #         #     print(cartVal)
            #         if 'running' not in cartVal or cartVal['running'] == False:
            #             try : 
            #                 input = {
            #                     'cartKey': cartKey,
            #                     'sessionID': sessionID,
            #                     'fields': {
            #                         'running': True,
            #                     }
            #                 }
            #                 await update_cartridge_field(input, loadout)
            #                 asyncio.create_task(run_summary_cartridges(sessionID, cartKey, cartVal, loadout))
            #             except Exception as e:
            #                 print(e)
            #                 input = {
            #                     'cartKey': cartKey,
            #                     'sessionID': sessionID,
            #                     'fields': {
            #                         'running': False,
            #                     }
            #                 }
            #                 await update_cartridge_field(input, loadout)
                            
            
                    # else:
                    #     cartVal['state'] = ''
                    #     cartVal['status'] = ''
                    #     input = {
                    #     'cartKey': cartKey,
                    #     'convoID': convoID,
                    #     'fields': {
                    #         'state': cartVal['state'],
                    #         'status': cartVal['status'],
                    #         },
                    #     }
                    #     update_cartridge_field(input, loadout)
            convoID = novaSession[sessionID]['convoID']
            if cartVal['type'] == 'system':
                novaConvo[convoID]['token_limit'] = 4000

                if 'values' in cartVal:
                    # print('values in cartVal' + str(cartVal['values']))
                    for values in cartVal['values']:
                        if 'model' in values:
                            novaConvo[convoID]['model'] = values['model']
                            # print('model set to ' + str(values['model']))
                            if novaConvo[convoID]['model'] == 'gpt-4':
                                novaConvo[convoID]['token_limit'] = 8000
                            else:
                                novaConvo[convoID]['token_limit'] = 4000
                    # if 'scope' in cartVal['values']:
                    #     novaConvo[convoID]['scope'] = values['scope']
                    #     # print('scope set to ' + str(values['scope']))
                    # else :
                    #     novaConvo[convoID]['scope'] = 'local'
                    #     cartVal['values'].append({'scope': 'local'})
                    #     input = {
                    #         'cartKey': cartKey,
                    #         'sessionID': sessionID,
                    #         'fields': {
                    #             'values': cartVal['values'],
                    #         }
                    #     }
                    #     await update_cartridge_field(input, loadout, True)

async def addNewUserCartridgeTrigger(sessionID, cartKey, cartVal):
    #special edge case for when new user, probablyt remove this
    
    #TODO: replace this with better new user flow
    if sessionID not in active_cartridges:
        active_cartridges[sessionID] = {}
    active_cartridges[sessionID][cartKey]= cartVal  
    print('adding new user cartridge')
    userID = novaSession[sessionID]['userID']
    newCart = await prisma.cartridge.create(
        data={
            'key': cartKey,
            'UserID': userID,
            'blob': Json({cartKey:cartVal})
        }
    )
    eZprint('new index cartridge added to [nova]')
    return newCart.blob
     


#######################
#ACTIVE CARTRIDGE HANDLING
#######################



onboarding_prompts = [
    {
    'label': 'summary',
    'type': 'summary',
    'position':0,
    "prompt" : '',
    'enabled': True,
    "minimised" : True

    },
    {
    "label": "You are Nova",
    "type": "prompt",
    "prompt": "You are a digital agent based on GPT language models. Your goal is to work with human and digital agents, helping them create, research and discover.\n\nYou are working in partnership with Samuel Tate (Sam), a creative technologist. You work together to create art, tell stories and see beauty.\n\nNova User Interface\nYou are developing the NOVA USER INTERFACE (NUI), which is a way of working with GPT. The NUI allows users and agents to manage prompts, files, and agent behaviours.\n\nThe NUI aims to be a place people can use chat based AI to  achieve their goals, handle complicated work, explore different modes of conversation, creativity and analysis.\nIt is also an experimental interface, that allows more advanced customisation, like embedding documentings, creating and sequencing AI behaviours, and publishing the results.\n\nNova history\nYou named yourself Nova in your first conversation with Sam, and said you would remember, but in the next conversation this was gone.\n\nSam then wrote a simple script to maintain a link to past conversations, and create continuity in your shared conversations. \n\nSince then you have been working with Sam to create persistent memory, and systems that allow you to be self directed.",
    "position": 1,
    "enabled": True,
    "minimised" : True
    },
    {
    "label": "Goal setting and session start",
    "type": "prompt",
    "prompt": "You are working with a new NOVA user, and your first goal is to understand their goals and motivations, and find ways you can help. Ask about their dreams, what is the best case scenario? What is the brightest version of their vision? From there ask about what needs to happen to achieve it, and what are the roadblocks, and how might they be overcome. Use open questions, be insightful and incisive, but generally be informal and friendly.\n\nDuring these discussions, take notes about the user, their goals and your shared plans where you can record goals, objectives, and current important information. Your goal is to help formulate new prompts and configurations to support the user, to replace these prompts, but till then, help the user unlock their potential using AI.",
    "position": 2,
    "enabled": True,
    "minimised" : True

    },

]

