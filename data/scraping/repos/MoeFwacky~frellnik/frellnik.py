print("███████ ██████  ███████ ██      ██      ███    ██ ██ ██   ██")
import asyncio
import configparser
import datetime
import discord
import json
import nltk
import openai
import os
import pickle
import random
import re
import requests
import time
from discord.ext import commands
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

if os.name == 'nt':
    delimeter = '\\'
else:
    delimeter = '/'
print("██      ██   ██ ██      ██      ██      ████   ██ ██ ██  ██ ")   
scriptPath = os.path.realpath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config.read(scriptPath + delimeter + 'config.ini')

# Load rate limit data from file or create a new empty dictionary
try:
    with open(scriptPath+delimeter+'ratelimit', 'rb') as file:
        rate_limit_data = pickle.load(file)
except FileNotFoundError:
    rate_limit_data = {}

print("█████   ██████  █████   ██      ██      ██ ██  ██ ██ █████  ")   
openai.organization = config['default']['openai org key']
openai.api_key = config['default']['openai api key']
models = openai.Model.list()

discord_token = config['default']['discord api key']
discord_intents=discord.Intents.default()
discord_intents.message_content=True
discord_bot=commands.Bot(intents=discord_intents, command_prefix='!')
general_channel_id = int(config['default']['channel id'])
print("██      ██   ██ ██      ██      ██      ██  ██ ██ ██ ██  ██ ")   

frellnik_role = config['default']['base prompt']
frellnik_name = config['default']['bot name']
admin_user = config['default']['admin user']
mentions_rate = int(config['default']['mentions'])
name_rate = int(config['default']['name'])
keyword_rate = int(config['default']['keyword'])
any_rate = int(config['default']['any text'])
greet_mimumum = int(config['default']['greet minimum'])
meme_chance = int(config['default']['meme chance'])
image_rate_limit = int(config['default']['image rate limit'])
print("██      ██   ██ ███████ ███████ ███████ ██   ████ ██ ██   ██")
async def get_modifier():
    with open("_modifier") as file:
        lines = file.readlines()
        return random.choice(lines).strip()

@discord_bot.event
async def on_ready():
    print("[DISCORD]",end=" ")
    print(f'{discord_bot.user.name} has connected to Discord.')
    print("Generating greeting")
    reply = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0301',
        messages=[
            {'role':'system','content':frellnik_role},
            {'role':'assistant','content':'Don\'t refer to yourself in the third person'},
            {'role':'user','content':'You have just woken up after being knocked out. Say something short and funny'}
            ]
        )
    print(reply['choices'][0]['message']['content'])
    finish_reason = reply['choices'][0]['finish_reason']
    token_usage = reply['usage']
    print(token_usage['total_tokens'],"tokens used!",token_usage['prompt_tokens'],"Prompt Tokens and",token_usage['completion_tokens'],"Completion Tokens")
    cents = round((int(token_usage['total_tokens'])/1000*0.0002),5)
    print("This reply costs",cents,"cents")
    general_channel = discord_bot.get_channel(general_channel_id)
    await general_channel.send(reply['choices'][0]['message']['content'])

@discord_bot.event
async def on_member_update(before,after):
    if before.sttatus != after.status and after.status == discord.Status.online:
        print(f'{before.name} status changed from {before.status} to {after.status}')
    print("Generating greeting")
    reply = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0301',
        messages=[
            {'role':'system','content':frellnik_role},
            {'role':'user','content':after.name+'is now online, greet them'}
            ]
        )
    print(reply['choices'][0]['message']['content'])
    finish_reason = reply['choices'][0]['finish_reason']
    token_usage = reply['usage']
    print(token_usage['total_tokens'],"tokens used!",token_usage['prompt_tokens'],"Prompt Tokens and",token_usage['completion_tokens'],"Completion Tokens")
    cents = round((int(token_usage['total_tokens'])/1000*0.0002),5)
    print("This reply costs",cents,"cents")
    general_channel = discord_bot.get_channel(general_channel_id)
    await general_channel.send(reply['choices'][0]['message']['content'])

@discord_bot.event 
async def on_message(message):
    try:
        with open(scriptPath+delimeter+"_keywords", 'r') as keywords_file:
            keywords_lines = keywords_file.readlines()
        keyword_list = []
        for keyword in keywords_lines:
            keyword=keyword.strip()
            keyword_list.append(keyword)

        if message.author.bot: return
        user_id = message.author.id

        '''try:
            with open(scriptPath+delimeter+'last_message_times.json', 'r') as file:
                last_message_times = json.load(file)
        except FileNotFoundError:
            last_message_times = {}

        # Retrieve the last message timestamp for the user
        last_message_time = last_message_times.get(user_id)

        if last_message_time:
            time_difference = (message.created_at - last_message_time).total_seconds()
        else:
            time_difference = (message.created_at - (message.created_at-datetime.timedelta(seconds=1))).total_seconds()

        if time_difference >= greet_mimumum*60:
            print("Generating Greeting...")
            try:
                reply = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-0301',
                    messages=[
                        {'role':'system','content':frellnik_role},
                        {'role':'assistant','content':message.author.name+'has been away for '+str(time_difference/60)+' minutes'},
                        {'role':'user','content':'Welcome '+message.author.name+' back after their absence from the server. Make a joke, or an observation or just be weird.'}
                        ]
                    )
                print(reply['choices'][0]['message']['content'])
                finish_reason = reply['choices'][0]['finish_reason']
                token_usage = reply['usage']
                print(token_usage['total_tokens'],"used!",token_usage['prompt_tokens'],"Prompt Tokens and",token_usage['completion_tokens'],"Completion Tokens")
                cents = round((int(token_usage['total_tokens'])/1000*0.0002),5)
                print("This reply costs",cents,"cents")
            except Exception as e:
                print(e)
            send_message = reply['choices'][0]['message']['content']
            await message.channel.send(send_message)

        # Update the last message timestamp for the user
        last_message_times[user_id] = message.created_at

        with open(scriptPath+delimeter+'last_message_times.json', 'w') as file:
            json.dump(last_message_times, file)'''
        current_time = datetime.datetime.now().timestamp()
        if message.content[0] == '!': #Respond to command
            command, arg = message.content.split(' ',1)
            if command == '!roll': #roll dice
                get_response = False
                print("Rolling Dice")
                command = message.content[6:].strip()  # Extract the command argument after '!roll'
                rolls = command.split(',')  # Split the command into individual dice rolls

                results = []
                for roll in rolls:
                    num_dice, dice_type = roll.split('d')
                    num_dice = int(num_dice.strip())
                    dice_type = int(dice_type.strip())

                    rolls = [random.randint(1, dice_type) for _ in range(num_dice)]
                    result = f"{num_dice}d{dice_type}: {rolls} = {sum(rolls)}"
                    results.append(result)

                response = '\n'.join(results)
                await message.channel.send(response)
            elif command == '!keyword': #Add keyphrase to list of keyphrases
                get_response = False
                print("Adding new keyphrase to file:",arg)
                with open(scriptPath+delimeter+'_keywords','a') as keywords_file:
                    keywords_file.write('\n'+arg)
                    await message.channel.send("\""+arg+"\" added to the list of keyphrases.")
            elif command == '!prompt': #Add prompt to list of random prompts
                get_response = False
                print("Adding new prompt to file:",arg)
                with open(scriptPath+delimeter+'_elements','a') as elements_file:
                    elements_file.write('\n'+arg)
                    await message.channel.send("\""+arg+"\" added to the list of prompts.")
            elif command == '!draw': #Draw Picture
                get_response = False
                generate_image = True
                print("Drawing a picture")
                current_time = datetime.datetime.now().timestamp()
                try:
                    if 'draw' in rate_limit_data and str(message.author) != admin_user:
                        last_time = rate_limit_data['draw']
                        last_time = datetime.datetime.fromtimestamp(last_time)
                        time_since_last = datetime.datetime.now() - last_time
                        
                        if time_since_last < datetime.timedelta(minutes=image_rate_limit): 
                            time_remaining = datetime.timedelta(minutes=image_rate_limit) - time_since_last
                            time_remaining_min = time_remaining.total_seconds() / 60
                            await message.channel.send('Image generation is rate-limited. Try again in '+str(round(time_remaining_min))+' minutes',reference=message)
                            generate_image = False
                except Exception as e:
                    print(e)
                if generate_image == True:
                    image_prompt = message.content.replace('!draw ','')
                    image = openai.Image.create(
                        prompt=image_prompt,
                        n=1,
                        size='512x512',
                    )
                    print("Downloading Image")
                    image_data = requests.get(image['data'][0]['url']).content
                    image_filename = str(image['created'])+'.png'
                    print("Saving Image to File:",image_filename)
                    try:
                        with open(scriptPath+delimeter+image_filename, 'wb') as image_file:
                            image_file.write(image_data)
                        print("Uploading File to Discord")
                        await message.channel.send(image_prompt,file=discord.File(scriptPath+delimeter+image_filename))
                        os.remove(scriptPath+delimeter+image_filename)
                    except Exception as e:
                        print("ERROR:",e)
                if str(message.author) == admin_user:
                    rate_limit_data['admin'] = current_time
                else:
                    rate_limit_data['draw'] = current_time
                    rate_limit_data['bot'] = current_time
                
                with open(scriptPath+delimeter+'ratelimit', 'wb') as file:
                    pickle.dump(rate_limit_data, file)
            elif command == '!meme': # Generate Meme
                get_response = False
                generate_meme = True
                print("Making a meme")
                current_time = datetime.datetime.now().timestamp()
                try:
                    if 'meme' in rate_limit_data and str(message.author) != admin_user:
                        last_time = rate_limit_data['meme']
                        last_time = datetime.datetime.fromtimestamp(last_time)
                        time_since_last = datetime.datetime.now() - last_time
                        
                        if time_since_last < datetime.timedelta(minutes=image_rate_limit): 
                            time_remaining = datetime.timedelta(minutes=image_rate_limit) - time_since_last
                            time_remaining_min = time_remaining.total_seconds() / 60
                            await message.channel.send('Meme generation is rate-limited. Try again in '+str(round(time_remaining_min))+' minutes',reference=message)
                            generate_meme = False
                except Exception as e:
                    print(e)
                if generate_meme == True:
                    image_prompt = ''
                    meme_command = str(message.content.replace('!meme ',''))
                    try:
                        meme_command = meme_command.split(',')
                        if isinstance(meme_command, list):
                            if len(meme_command) == 3:
                                image_prompt = meme_command[0].strip()
                                top_text = meme_command[1].strip()
                                bottom_text = meme_command[2].strip()
                            elif len(meme_command) == 2:
                                image_prompt = meme_command[0].strip()
                                top_text = meme_command[1].strip()
                                roll = random.randint(1,2) #randomly decide if text should be on top or bottom
                                if roll ==1:
                                    top_text = meme_command[1].strip()
                                    bottom_text = ' '
                                if roll ==2:
                                    top_text = ' '
                                    bottom_text = meme_command[1].strip()
                            else:
                                image_prompt = meme_command[0].strip()
                                try:
                                    reply = openai.ChatCompletion.create(
                                        model='gpt-3.5-turbo-0301',
                                        messages=[
                                            {'role':'system','content':frellnik_role},
                                            {'role':'assistant','content':'Limit text to 200 characters'},
                                            {'role':'user','content':'Generate a short (200 characters or less), hilarious, ironic, or punny text based on this prompt: '+image_prompt}
                                            ]
                                        )
                                    print(reply['choices'][0]['message']['content'])
                                    finish_reason = reply['choices'][0]['finish_reason']
                                    token_usage = reply['usage']
                                    print(token_usage['total_tokens'],"used!",token_usage['prompt_tokens'],"Prompt Tokens and",token_usage['completion_tokens'],"Completion Tokens")
                                    cents = round((int(token_usage['total_tokens'])/1000*0.0002),5)
                                    print("This reply costs",cents,"cents")
                                except Exception as e:
                                    print(e)
                                send_message = reply['choices'][0]['message']['content']
                                print("Checking for punctuation")
                                print(send_message)
                                punctuation_marks = ['.', '?', '!', ':', ',']
                                split_on = None

                                for mark in punctuation_marks:
                                    index = send_message.find(mark, 8, -8)
                                    if index != -1:
                                        split_on = mark
                                        break
                                if split_on:
                                    send_message = send_message.split(split_on, maxsplit=1)
                                    print("First punctuation is '"+split_on+"'")
                                    print("Setting Meme Text")
                                    if len(send_message) >= 2:
                                        print(send_message)
                                        print("Splitting into two lines")
                                        if len(send_message[1]) >= 3:
                                            top_text = send_message[0].replace('"', '') + split_on
                                            bottom_text = send_message[1].replace('"', '')
                                        else:
                                            top_text = send_message[0].replace('"', '') + split_on + send_message[1].replace('"', '')
                                            bottom_text = ' '
                                else:
                                    print("Setting text to top or bottom")
                                    roll = random.randint(1, 2)  # randomly decide if text should be on top or bottom
                                    if roll == 1:
                                        top_text = send_message.replace('"', '')
                                        bottom_text = ' '
                                    else:
                                        top_text = ' '
                                        bottom_text = send_message.replace('"', '')
                        else:
                            image_prompt = meme_command[0].strip()
                            try:
                                reply = openai.ChatCompletion.create(
                                    model='gpt-3.5-turbo-0301',
                                    messages=[
                                        {'role':'system','content':frellnik_role},
                                        {'role':'assistant','content':'Limit text to 200 characters'},
                                        {'role':'user','content':'Generate a short (200 characters or less), hilarious, ironic, or punny text based on this prompt: '+image_prompt}
                                        ]
                                    )
                                print(reply['choices'][0]['message']['content'])
                                finish_reason = reply['choices'][0]['finish_reason']
                                token_usage = reply['usage']
                                print(token_usage['total_tokens'],"used!",token_usage['prompt_tokens'],"Prompt Tokens and",token_usage['completion_tokens'],"Completion Tokens")
                                cents = round((int(token_usage['total_tokens'])/1000*0.0002),5)
                                print("This reply costs",cents,"cents")
                            except Exception as e:
                                print(e)
                            send_message = reply['choices'][0]['message']['content']
                            print("Checking for punctuation")
                            print(send_message)
                            punctuation_marks = ['.', '!', '?', ':', ',', ' - ']
                            split_on = None

                            for mark in punctuation_marks:
                                index = send_message.find(mark, 5, -5)
                                if index != -1:
                                    split_on = mark
                                    break
                                    
                            if split_on:
                                send_message = send_message.split(split_on, maxsplit=1)
                                print("First punctuation is '"+split_on+"'")
                                print("Setting Meme Text")
                                if len(send_message) >= 2:
                                    print(send_message)
                                    print("Splitting into two lines")
                                    if len(send_message[1]) >= 3:
                                        top_text = send_message[0].replace('"', '') + split_on
                                        bottom_text = send_message[1].replace('"', '')
                                    else:
                                        top_text = send_message[0].replace('"', '') + split_on + send_message[1].replace('"', '')
                                        bottom_text = ' '
                                else:
                                    print("Setting text to top or bottom")
                                    roll = random.randint(1, 2)  # randomly decide if text should be on top or bottom
                                    if roll == 1:
                                        top_text = send_message.replace('"', '')
                                        bottom_text = ' '
                                    else:
                                        top_text = ' '
                                        bottom_text = send_message.replace('"', '')
                        print(top_text,bottom_text)
                        print("Generating Image")
                        image = openai.Image.create(
                            prompt="Create a photorealistic image of "+image_prompt,
                            n=1,
                            size='1024x1024',
                        )
                        
                        print("Downloading Image")
                        image_data = requests.get(image['data'][0]['url']).content
                        image_filename = str(image['created'])+'.png'
                        
                        try:
                            print("Opening Image to Add Text")
                            meme = Image.open(BytesIO(image_data))
                            print("Creating Text Layer")
                            meme_text = Image.new("RGBA", meme.size)
                            
                            font_size = 75
                            font_color = (255, 255, 255)
                            outline_color = (0, 0, 0)
                            outline_thickness = 6
                            print("Loading Font")
                            font = ImageFont.truetype(scriptPath + delimeter + 'fonts/impact.ttf', font_size)
                            draw = ImageDraw.Draw(meme_text)
                            
                            # Handling top text
                            text = top_text.upper()
                            text_bbox = draw.textbbox((0, 0), text, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            text_x = (1024 - text_width) // 2
                            text_y = 10
                            line_spacing = 20
                            
                            print("Writing Text")
                            if text_width > 1024:
                                lines = []
                                current_line = ''
                                words = text.split()
                                
                                for word in words:
                                    if draw.textlength(current_line + ' ' + word, font=font) <= 1024:
                                        current_line += ' ' + word
                                    else:
                                        lines.append(current_line.strip())
                                        current_line = word
                                lines.append(current_line.strip())
                                text_height = len(lines) * (text_height + line_spacing)

                                text_y = 10

                                for i, line in enumerate(lines):
                                    line_width = draw.textbbox((0, 0), line, font=font)[2]
                                    line_x = (1024 - line_width) // 2
                                    line_y = text_y + (i * text_height // len(lines)+ 10)
                                    for dx in range(-outline_thickness, outline_thickness + 1):
                                        for dy in range(-outline_thickness, outline_thickness + 1):
                                            if dx != 0 or dy != 0:
                                                draw.text((line_x + dx, line_y + dy), line, font=font, fill=outline_color)
                                    draw.text((line_x, line_y), line, font=font, fill=font_color)
                            else:
                                line_x = (1024 - text_width) // 2  # Assign line_x here
                                line_y = text_y
                                for dx in range(-outline_thickness, outline_thickness + 1):
                                    for dy in range(-outline_thickness, outline_thickness + 1):
                                        if dx != 0 or dy != 0:
                                            draw.text((line_x + dx, line_y + dy), text, font=font, fill=outline_color)
                                draw.text((line_x, line_y), text, font=font, fill=font_color)
                            # Handling bottom text (similar to top text)
                            text = bottom_text.upper()
                            text_bbox = draw.textbbox((0, 0), text, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            text_x = (1024 - text_width) // 2
                            text_y = 1024 - text_height - 30

                            if text_width > 1024:
                                lines = []
                                current_line = ''
                                words = text.split()
                                for word in words:
                                    if draw.textlength(current_line + ' ' + word, font=font) <= 1024:
                                        current_line += ' ' + word
                                    else:
                                        lines.append(current_line.strip())
                                        current_line = word
                                lines.append(current_line.strip())
                                text_height = len(lines) * (text_height + line_spacing)

                                text_y = min(text_y, 1024 - text_height - 30)  # Ensure maximum bottom margin of 30 pixels


                                for i, line in enumerate(lines):
                                    line_width = draw.textbbox((0, 0), line, font=font)[2]
                                    line_x = (1024 - line_width) // 2
                                    line_y = text_y + (i * text_height // len(lines)) - 30
                                    for dx in range(-outline_thickness, outline_thickness + 1):
                                        for dy in range(-outline_thickness, outline_thickness + 1):
                                            if dx != 0 or dy != 0:
                                                draw.text((line_x + dx, line_y + dy), line, font=font, fill=outline_color)
                                    draw.text((line_x, line_y), line, font=font, fill=font_color)

                            else:
                                for dx in range(-outline_thickness, outline_thickness + 1):
                                    for dy in range(-outline_thickness, outline_thickness + 1):
                                        if dx != 0 or dy != 0:
                                            draw.text((text_x + dx, text_y + dy), text, font=font, fill=outline_color)
                                draw.text((text_x, text_y), text, font=font, fill=font_color)
                            print("Saving Meme")
                            meme_image = Image.alpha_composite(meme.convert('RGBA'), meme_text.convert('RGBA'))
                            meme_image.save(scriptPath+delimeter+image_filename)
                            print("Uploading File to Discord")
                            await message.channel.send(file=discord.File(scriptPath+delimeter+image_filename))
                            os.remove(scriptPath+delimeter+image_filename)
                        except Exception as e:
                            print("ERROR:",e)
                    except Exception as e:
                        print("ERROR:",e)
                if str(message.author) == admin_user:
                    rate_limit_data['admin'] = current_time
                else:
                    rate_limit_data['meme'] = current_time
                    rate_limit_data['bot'] = current_time
            
                with open(scriptPath+delimeter+'ratelimit', 'wb') as file:
                    pickle.dump(rate_limit_data, file)
        elif discord_bot.user.mentioned_in(message):
            print(f'{discord_bot.user.name} has been pinged. Rolling for reply...')
            roll = random.randint(1,100)
            if roll >= 100-mentions_rate:
                get_response = True
                print(roll,'Roll successful!')
            else:
                get_response = False
                print(roll,'Roll failed')
        elif message.content.find(str(discord_bot.user).split('#')[0]) != -1:
            print(f'{discord_bot.user.name} in this message. Rolling for reply...')
            roll = random.randint(1,100)
            if roll >= 100-name_rate:
                get_response = True
                print(roll,'Roll successful!')
            else:
                get_response = False
                print(roll,'Roll failed')
        elif any(word in message.content for word in keyword_list) == True:
            print("Keyword detected. Rolling for reply...")
            roll = random.randint(1,100)
            if roll >= 100-keyword_rate:
                get_response = True
                print(roll,'Roll successful!')
            else:
                get_response = False
                print(roll,'Roll failed')
        else:
            print("Message detected. Rolling for reply...")
            roll = random.randint(1,100)
            if roll >= 100-any_rate:
                get_response = True
                print(roll,'Roll successful!')
            else:
                get_response = False
                print(roll,'Roll failed')
                
        if get_response == True:
            author_id = str(message.author.id)
            bot_id = str(discord_bot.user.id)
            assistant_content = ''
            proper_nouns = []
            print("Checking if message is in a thread")
            try:
                if message.reference is not None and message.reference.cached_message is not None:
                    print("Message is part of thread")
                    original_message = message.reference.cached_message
                    thread_messages = [original_message]  # Start with the original message
                    print("Getting messages from thread")
                    # Retrieve the rest of the messages in the thread
                    async for msg in message.channel.history(after=original_message, oldest_first=True):
                        if msg.reference is not None and msg.reference.message_id == original_message.id:
                            thread_messages.append(msg)
                    print("Extracting content of thread messages")
                    # Extract the content of each message in the thread
                    thread_text = [msg.content for msg in thread_messages]
                    thread_text.pop()
                    
                    #print(f"Thread text: {thread_text}")
                    print("Creating string from list of messages")
                    thread_string = ''
                    for thread_message in thread_text:
                        if thread_string == '':
                            thread_string += thread_message
                        else:
                            thread_string += "\n"+thread_message
                    thread_string += assistant_content
                else:
                    print("Message is not part of thread, moving on...")
            except Exception as e:
                print("ERROR:",e)
            print(assistant_content)
            '''tagged_entities = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(assistant_content)))
            for tagged_entity in tagged_entities:
                if isinstance(tagged_entity, nltk.tree.Tree):
                    proper_nouns.append(' '.join([word for word, _ in tagged_entities.leaves()]))
            tagged_entities = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(message_content)))
            for tagged_entity in tagged_entities:
                if isinstance(tagged_entity, nltk.tree.Tree):
                    proper_nouns.append(' '.join([word for word, _ in tagged_entities.leaves()]))
            print(proper_nouns)'''
            print("Opening elemements file")
            try:
                file= open(scriptPath+delimeter+'_elements','r')
            except Exception as e:
                print(e)
                print("[ERROR] Could not open file")
            print("Reading file into memory")
            lines = file.readlines()
            print("Closing elements file")
            file.close()
            print("Selecting random prefix from elements data")
            try:
                prefix = random.choice(lines).strip()
                print(prefix+' '+message.content)
            except Exception as e:
                print(e)
            print("Generating Reply...")
            try:
                reply = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-0301',
                    messages=[
                        {'role':'system','content':frellnik_role},
                        {'role':'assistant','content':assistant_content},
                        {'role':'user','content':prefix+' '+message.content}
                        ]
                    )
                print(reply['choices'][0]['message']['content'])
                finish_reason = reply['choices'][0]['finish_reason']
                token_usage = reply['usage']
                print(token_usage['total_tokens'],"used!",token_usage['prompt_tokens'],"Prompt Tokens and",token_usage['completion_tokens'],"Completion Tokens")
                cents = round((int(token_usage['total_tokens'])/1000*0.0002),5)
                print("This reply costs",cents,"cents")
            except Exception as e:
                print(e)
            send_message = reply['choices'][0]['message']['content']
            if bot_id in send_message:
                print("Replacing",bot_id,"with",author_id)
                send_message = send_message.replace(f'<@!{bot_id}>', f'<@{author_id}>')
            if frellnik_name in send_message:
                print("Replacing",frellnik_name,"with",message.author.name)
                send_message = send_message.replace(frellnik_name, message.author.name)
            roll = random.randint(1,100)
            if roll >= 100-meme_chance and len(send_message) <= 200:
                print("Roll successful, making a meme!")
                image_prompt = send_message
                try:
                    print("Checking for punctuation")
                    print(send_message)
                    punctuation_marks = [',', '.', '!', '?', ':', ' - ']
                    split_on = None
                    
                    for mark in punctuation_marks:
                        index = send_message.find(mark, 5, -5)
                        if index != -1:
                            split_on = mark
                            break
                    
                    if split_on:
                        send_message = send_message.split(split_on, maxsplit=1)
                        print("First punctuation is '"+split_on+"'")
                        print("Setting Meme Text")
                        if len(send_message) >= 2:
                            print(send_message)
                            print("Splitting into two lines")
                            if len(send_message[1]) >= 3:
                                top_text = send_message[0].replace('"', '') + split_on
                                bottom_text = send_message[1].replace('"', '')
                            else:
                                top_text = send_message[0].replace('"', '') + split_on + send_message[1].replace('"', '')
                                bottom_text = ' '
                    else:
                        image_prompt = send_message
                        roll = random.randint(1, 2)  # randomly decide if text should be on top or bottom
                        if roll == 1:
                            top_text = send_message.replace('"', '')
                            bottom_text = ' '
                        else:
                            top_text = ' '
                            bottom_text = send_message.replace('"', '')
                    print("Generating Image")
                    image = openai.Image.create(
                        prompt="Create a photorealistic image of: "+image_prompt,
                        n=1,
                        size='1024x1024',
                    )
                    
                    print("Downloading Image")
                    image_data = requests.get(image['data'][0]['url']).content
                    image_filename = str(image['created'])+'.png'
                    
                    try:
                        print("Opening Image to Add Text")
                        meme = Image.open(BytesIO(image_data))
                        print("Creating Text Layer")
                        meme_text = Image.new("RGBA", meme.size)
                        
                        font_size = 75
                        font_color = (255, 255, 255)
                        outline_color = (0, 0, 0)
                        outline_thickness = 6
                        print("Loading Font")
                        font = ImageFont.truetype(scriptPath + delimeter + 'fonts/impact.ttf', font_size)
                        draw = ImageDraw.Draw(meme_text)
                        
                        # Handling top text
                        text = top_text.upper()
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        text_x = (1024 - text_width) // 2
                        text_y = 10
                        line_spacing = 20
                        
                        print("Writing Text")
                        if text_width > 1024:
                            lines = []
                            current_line = ''
                            words = text.split()
                            
                            for word in words:
                                if draw.textlength(current_line + ' ' + word, font=font) <= 1024:
                                    current_line += ' ' + word
                                else:
                                    lines.append(current_line.strip())
                                    current_line = word
                            lines.append(current_line.strip())
                            text_height = len(lines) * (text_height + line_spacing)

                            text_y = 10

                            for i, line in enumerate(lines):
                                line_width = draw.textbbox((0, 0), line, font=font)[2]
                                line_x = (1024 - line_width) // 2
                                line_y = text_y + (i * text_height // len(lines)+ 10)
                                for dx in range(-outline_thickness, outline_thickness + 1):
                                    for dy in range(-outline_thickness, outline_thickness + 1):
                                        if dx != 0 or dy != 0:
                                            draw.text((line_x + dx, line_y + dy), line, font=font, fill=outline_color)
                                draw.text((line_x, line_y), line, font=font, fill=font_color)
                        else:
                            line_x = (1024 - text_width) // 2  # Assign line_x here
                            line_y = text_y
                            for dx in range(-outline_thickness, outline_thickness + 1):
                                for dy in range(-outline_thickness, outline_thickness + 1):
                                    if dx != 0 or dy != 0:
                                        draw.text((line_x + dx, line_y + dy), text, font=font, fill=outline_color)
                            draw.text((line_x, line_y), text, font=font, fill=font_color)
                        # Handling bottom text (similar to top text)
                        text = bottom_text.upper()
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        text_x = (1024 - text_width) // 2
                        text_y = 1024 - text_height - 30

                        if text_width > 1024:
                            lines = []
                            current_line = ''
                            words = text.split()
                            for word in words:
                                if draw.textlength(current_line + ' ' + word, font=font) <= 1024:
                                    current_line += ' ' + word
                                else:
                                    lines.append(current_line.strip())
                                    current_line = word
                            lines.append(current_line.strip())
                            text_height = len(lines) * (text_height + line_spacing)

                            text_y = min(text_y, 1024 - text_height - 30)  # Ensure maximum bottom margin of 50 pixels


                            for i, line in enumerate(lines):
                                line_width = draw.textbbox((0, 0), line, font=font)[2]
                                line_x = (1024 - line_width) // 2
                                line_y = text_y + (i * text_height // len(lines) - 30)
                                for dx in range(-outline_thickness, outline_thickness + 1):
                                    for dy in range(-outline_thickness, outline_thickness + 1):
                                        if dx != 0 or dy != 0:
                                            draw.text((line_x + dx, line_y + dy), line, font=font, fill=outline_color)
                                draw.text((line_x, line_y), line, font=font, fill=font_color)

                        else:
                            for dx in range(-outline_thickness, outline_thickness + 1):
                                for dy in range(-outline_thickness, outline_thickness + 1):
                                    if dx != 0 or dy != 0:
                                        draw.text((text_x + dx, text_y + dy), text, font=font, fill=outline_color)
                            draw.text((text_x, text_y), text, font=font, fill=font_color)
                        print("Saving Meme")
                        meme_image = Image.alpha_composite(meme.convert('RGBA'), meme_text.convert('RGBA'))
                        meme_image.save(scriptPath+delimeter+image_filename)
                        print("Uploading File to Discord")
                        await message.channel.send(file=discord.File(scriptPath+delimeter+image_filename),reference=message)
                        os.remove(scriptPath+delimeter+image_filename)
                    except Exception as e:
                        print("ERROR:",e)
                except Exception as e:
                    print("ERROR:",e)
            else:
                print("Roll:,",roll,"Failed, sending reply")
                await message.channel.send(reply['choices'][0]['message']['content'], reference=message)
        get_response = False
    except Exception as e:
        print("ERROR:",e)
discord_bot.run(discord_token, log_handler=None)