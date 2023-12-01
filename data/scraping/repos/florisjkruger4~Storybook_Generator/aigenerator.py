import os
import openai
from django.conf import settings
from random import randint
from gtts import gTTS
from Storybook_Generator.settings import STATIC_URL

openai.api_key =  settings.OPENAI_API_KEY
openai.Model.list()

audioFilePath = os.path.join(STATIC_URL, 'audio/StoryAudio.mp3')

def returnStorybookName(bookName):
    response = openai.Completion.create(
        model="text-davinci-001",
        prompt="Generate the name for a childrens storybook about \n {}".format(bookName),
        temperature=0.25,
        max_tokens=15,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
    )

    if 'choices' in response:
        if len(response['choices'])>0:
            answer = response['choices'][0]['text'].replace('\n', '')
            return answer
        else:
            return ''
    else:
        return ''

def returnStorybookImg(bookImg):
    response = openai.Image.create(
        prompt = "Generate a kids storybook cover image of \n {}".format(bookImg) + " in a cartoon style.",
        n = 1,
        size = "256x256"
    )

    imageOne_url = response['data'][0]['url']

    return imageOne_url


# Fantasy Storybook Generator 
def returnFantasyStorybookStory(name, quest, location):

    randomPromptSelect = randint(0, 2)
    fantasyPromptsCompleted = []

    firstfantasyPrompts1value = randint(0, 6)
    firstfantasyPrompts1 = ["funny", "heroic", "courageous", "daunting", "impactful", "happy", "magical"]
    firstfantasyPrompts2value = randint(0, 1)
    firstfantasyPrompts2 = ["person", "hero"]
    fantasyPromptsCompleted.append("Generate a lengthy and complete " + firstfantasyPrompts1[firstfantasyPrompts1value] + " fantasy story for a child about a " + firstfantasyPrompts2[firstfantasyPrompts2value] + " protagonist named {}".format(name) + " who is on a quest {}".format(quest) + " that takes place in or on {}".format(location) + ". The story should have some sort of lesson to share.")

    secondfantasyPrompts1value = randint(0, 1)
    secondfantasyPrompts1 = ["past", "future"]
    secondfantasyPrompts2value = randint(0, 3)
    secondfantasyPrompts2 = ["sunny", "cold", "stormy", "cloudy"]
    secondfantasyPrompts3value = randint(0, 1)
    secondfantasyPrompts3 = ["person", "hero"]
    secondfantasyPrompts4value = randint(0, 5)
    secondfantasyPrompts4 = ["goblins", "trolls", "witches", "aliens", "robots", "pirates"]
    fantasyPromptsCompleted.append("Generate a lengthy and complete fantasy story for a child based in the " + secondfantasyPrompts1[secondfantasyPrompts1value] + " that takes place in or on {}".format(location) + " on a " + secondfantasyPrompts2[secondfantasyPrompts2value] + " day about a " + secondfantasyPrompts3[secondfantasyPrompts3value] + " protagonist named {}".format(name) + " who is on a mission {}".format(quest) + " with a trusty companion and encounters bad " + secondfantasyPrompts4[secondfantasyPrompts4value] + " foes along the way. The story should share some sort of message.")

    thirdfantasyPrompts1value = randint(0, 6)
    thirdfantasyPrompts1 = ["funny", "heroic", "courageous", "daunting", "impactful", "happy", "magical"]
    thirdfantasyPrompts2value = randint(0, 1)
    thirdfantasyPrompts2 = ["person", "hero"]
    thirdfantasyPrompts3value = randint(0, 2)
    thirdfantasyPrompts3 = ["loyal", "brave", "funny"]
    fantasyPromptsCompleted.append("Generate a lengthy and complete" + thirdfantasyPrompts1[thirdfantasyPrompts1value] + " fantasy story about a " + thirdfantasyPrompts2[thirdfantasyPrompts2value] + " protagonist named {}".format(name) + " who makes a group of " + thirdfantasyPrompts3[thirdfantasyPrompts3value] + " friends to go {}".format(quest) + " together and learns a valuable lesson and takes place in or on {}.".format(location))

    print(fantasyPromptsCompleted[randomPromptSelect])

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=fantasyPromptsCompleted[randomPromptSelect],
        temperature=0.35,
        max_tokens=4000,
        top_p=1,
        n=1,
        frequency_penalty=1,
        presence_penalty=1
    )

    if 'choices' in response:
        if len(response['choices'])>0:

            answer = response['choices'][0]['text'].replace('\n', '')

            print("please wait...PROSSESING AUDIO")
            audio = gTTS(text=answer, lang='en', tld='co.za', slow=False)

            # Save to mp3 in audio dir.
            audio.save(audioFilePath)
            
            return answer
        else:
            return ''
    else:
        return ''



# Sports Storybook Generator 
def returnSportsStorybookStory(name, sport, location):

    randomPromptSelect = randint(0, 2)
    sportsPromptsCompleted = []

    firstsportsPrompts1value = randint(0, 7)
    firstsportsPrompts1 = ["funny", "courageous", "daunting", "impactful", "happy", "heartwarming", "silly", "underdog"]
    firstsportsPrompts2value = randint(0, 1)
    firstsportsPrompts2 = ["outgoing", "shy"]
    sportsPromptsCompleted.append("Generate a lengthy and complete " + firstsportsPrompts1[firstsportsPrompts1value] + " sports story about a person named {}".format(name) + " who is an " + firstsportsPrompts2[firstsportsPrompts2value] + " player on a {}".format(sport) + " team who works together with teammates to win a tournament based in or on {}".format(location) + " in the form of a storybook for a child. The story should have some sort of moral or lesson and have a happy ending.")

    secondsportsPrompts1value = randint(0, 7)
    secondsportsPrompts1 = ["funny", "courageous", "daunting", "impactful", "happy", "heartwarming", "silly", "underdog"]
    secondsportsPrompts2value = randint(0, 3)
    secondsportsPrompts2 = ["tall", "short", "strong", "fast"]
    secondsportsPrompts3value = randint(0, 1)
    secondsportsPrompts3 = ["outgoing", "shy"]
    secondsportsPrompts4value = randint(0, 5)
    secondsportsPrompts4 = ["stormy", "rainy", "cold", "hot", "windy", "foggy"]
    sportsPromptsCompleted.append("Generate a lengthy and complete " + secondsportsPrompts1[secondsportsPrompts1value] + " sports story for a child about a " + secondsportsPrompts2[secondsportsPrompts2value] + " and " + secondsportsPrompts3[secondsportsPrompts3value] + " and determined person named {}".format(name) + " who forms a group of friends who decide to make flyers to start a new {}".format(sport) + " team to go on and play against a rival team on a " + secondsportsPrompts4[secondsportsPrompts4value] + " day based in or on {}".format(location) + ". The story should try to share a valuable lesson.")

    thirdsportsPrompts1value = randint(0, 7)
    thirdsportsPrompts1 = ["funny", "courageous", "daunting", "impactful", "happy", "heartwarming", "silly", "underdog"]
    thirdsportsPrompts2value = randint(0, 1)
    thirdsportsPrompts2 = ["outgoing", "shy"]
    sportsPromptsCompleted.append("Generate a lengthy and complete" + thirdsportsPrompts1[thirdsportsPrompts1value] + " sports story for a child based in or on {}".format(location) + " about a " + thirdsportsPrompts2[thirdsportsPrompts2value] + " person named {}".format(name) + " who works hard to get a spot on the team for a {}".format(sport) + " league and makes friends with a bully on the team and work together to win. The story should have some sort of moral or lesson to share.")

    print(sportsPromptsCompleted[randomPromptSelect])

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=sportsPromptsCompleted[randomPromptSelect],
        temperature=0.35,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
    )

    if 'choices' in response:
        if len(response['choices'])>0:

            answer = response['choices'][0]['text'].replace('\n', '')

            print("please wait...PROSSESING AUDIO")
            audio = gTTS(text=answer, lang='en', tld='co.za', slow=False)

            # Save to mp3 in audio dir.
            audio.save(audioFilePath)

            return answer
        else:
            return ''
    else:
        return ''



# Adventure Storybook Generator 
def returnAdventureStorybookStory(name, adventure, location):

    randomPromptSelect = randint(0, 2)
    adventurePromptsCompleted = []

    firstadventurePrompts1value = randint(0, 6)
    firstadventurePrompts1 = ["daring", "exciting", "dauntless", "stunning", "silly", "amazing", "breathtaking"] 
    firstadventurePrompts2value = randint(0, 3)
    firstadventurePrompts2 =  ["outgoing", "brave", "fearless", "wise"]
    adventurePromptsCompleted.append("Generate a lengthy and complete " + firstadventurePrompts1[firstadventurePrompts1value] + " adventure story about a " + firstadventurePrompts2[firstadventurePrompts2value] + " protagonist named {}".format(name) + "  who is on an adventure {}".format(adventure) + ". The story should be based in or on {}".format(location) + " in the form of a bedtime storybook for a child with some sort of lesson to share.")

    secondadventurePrompts1value = randint(0, 6)
    secondadventurePrompts1 = ["daring", "exciting", "dauntless", "stunning", "silly", "amazing", "breathtaking"] 
    secondadventurePrompts2value = randint(0, 3)
    secondadventurePrompts2 =  ["outgoing", "brave", "fearless", "wise"]
    adventurePromptsCompleted.append("Generate a lengthy and complete " + secondadventurePrompts1[secondadventurePrompts1value] + " adventure story for a child about an " + secondadventurePrompts2[secondadventurePrompts2value] + " protagonist named {}".format(name) + "  who is on an adventure {}".format(adventure) + " in or on {}".format(location) + ". The story should involve meeting a new friend to help them on their adventure and the story should have some sort of message.")

    thirdadventurePrompts1value = randint(0, 6)
    thirdadventurePrompts1 = ["daring", "exciting", "dauntless", "stunning", "silly", "amazing", "breathtaking"] 
    thirdadventurePrompts2value = randint(0, 3)
    thirdadventurePrompts2 =  ["outgoing", "brave", "courageous", "wise"]
    thirdadventurePrompts3value = randint(0, 2)
    thirdadventurePrompts3 =  ["fear", "conflict", "problme"]
    adventurePromptsCompleted.append("Generate a lengthy and complete " + thirdadventurePrompts1[thirdadventurePrompts1value] + " adventure story for a child about an " + thirdadventurePrompts2[thirdadventurePrompts2value] + " protagonist named {}".format(name) + "  who is on an adventure {}".format(adventure) + " in or on {}".format(location) + ". The story should involve the protagonist facing a " + thirdadventurePrompts3[thirdadventurePrompts3value] + " and overcoming it on their adventure and the story should have some sort of message.")

    print(adventurePromptsCompleted[randomPromptSelect])

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=adventurePromptsCompleted[randomPromptSelect],
        temperature=0.35,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
    )

    if 'choices' in response:
        if len(response['choices'])>0:

            answer = response['choices'][0]['text'].replace('\n', '')

            print("please wait...PROSSESING AUDIO")
            audio = gTTS(text=answer, lang='en', tld='co.za', slow=False)

            # Save to mp3 in audio dir.
            audio.save(audioFilePath)

            return answer
        else:
            return ''
    else:
        return ''



# Spooky Storybook Generator 
def returnSpookyStorybookStory(name, spooky, location):

    randomPromptSelect = randint(0, 2)
    spookyPromptsCompleted = []

    firstspookyPrompts1value = randint(0, 6)
    firstspookyPrompts1 = ["chilling", "scary", "intimidating", "frightening", "silly", "eerie", "breathtaking"] 
    firstspookyPrompts2value = randint(0, 3)
    firstspookyPrompts2 =  ["courageous", "brave", "curious", "timid"]
    spookyPromptsCompleted.append("Generate a lengthy and complete " + firstspookyPrompts1[firstspookyPrompts1value] + " spooky story for a child about a " + firstspookyPrompts2[firstspookyPrompts2value] + " protagonist named {}".format(name) + " who is {}".format(spooky) + " in or on {}".format(location) + ". The story should try and scare the reader and maybe even end on a cliffhanger.")

    secondspookyPrompts1value = randint(0, 6)
    secondspookyPrompts1 = ["chilling", "scary", "intimidating", "frightening", "silly", "eerie", "breathtaking"] 
    secondspookyPrompts2value = randint(0, 3)
    secondspookyPrompts2 =  ["courageous", "brave", "curious", "timid"]
    secondspookyPrompts3value = randint(0, 9)
    secondspookyPrompts3 = ["spiders", "snakes", "monsters", "vampires", "werewolves", "witches", "zombies", "skeletons", "ghouls", "ghosts"] 
    spookyPromptsCompleted.append("Generate a lengthy and complete " + secondspookyPrompts1[secondspookyPrompts1value] + " spooky story for a child about a " + secondspookyPrompts2[secondspookyPrompts2value] + " protagonist named {}".format(name) + " who is {}".format(spooky) + " in or on {}".format(location) + ". The story should involve the protagonist facing their fear of " + secondspookyPrompts3[secondspookyPrompts3value] + " and overcoming them with bravery and wit.")

    thirdspookyPrompts1value = randint(0, 6)
    thirdspookyPrompts1 = ["chilling", "scary", "intimidating", "frightening", "silly", "eerie", "breathtaking"] 
    thirdspookyPrompts2value = randint(0, 3)
    thirdspookyPrompts2 =  ["courageous", "brave", "curious", "timid"]
    thirdspookyPrompts3value = randint(0, 3)
    thirdspookyPrompts3 = ["cliffhanger", "happy", "mysterious", "suspenseful"]
    spookyPromptsCompleted.append("Generate a lengthy and complete " + thirdspookyPrompts1[thirdspookyPrompts1value] + " spooky story for a child about a " + thirdspookyPrompts2[thirdspookyPrompts2value] + " protagonist named {}".format(name) + " and a loyal companion. Both characters are {}".format(spooky) + " in or on {}".format(location) + ". The story should have a " + thirdspookyPrompts3[thirdspookyPrompts3value] + " ending.")

    print(spookyPromptsCompleted[randomPromptSelect])

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=spookyPromptsCompleted[randomPromptSelect],
        temperature=0.35,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
    )

    if 'choices' in response:
        if len(response['choices'])>0:

            answer = response['choices'][0]['text'].replace('\n', '')

            print("please wait...PROSSESING AUDIO")
            audio = gTTS(text=answer, lang='en', tld='co.za', slow=False)

            # Save to mp3 in audio dir.
            audio.save(audioFilePath)
            
            return answer
        else:
            return ''
    else:
        return ''



