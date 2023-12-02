
from datetime import date
from inspect import trace
import openai
from logging import exception
from operator import mul
from os import name, stat
from typing import AsyncContextManager, final
import discord
from discord import errors
from discord import client
from discord import channel
from discord import embeds
from discord.embeds import Embed
from discord.ext import commands
from discord.ext.commands.core import command
from discord.member import Member
from discord.player import PCMAudio
from discord.utils import time_snowflake
from openai.api_resources import model
from pymongo import MongoClient
import names
from pymongo.collection import _FIND_AND_MODIFY_DOC_FIELDS
import re
import random
import math
import asyncio
import linecache
import sys
import traceback
import string
import itertools
from imdb import IMDb
from pymongo.database import Database
from youtube_search import YoutubeSearch
import json
import youtube_dl
from discord_components import DiscordComponents, Button, ButtonStyle, InteractionType
import text2emotion as te
from removebg import RemoveBg
import os
from PIL import Image
from io import BytesIO
import requests
import Globals
import pymongo
import ssl

class noImageError(commands.CommandError):
    def __init__(self, user, *args, **kwargs):
        self.user = user

def getMongo():
    return MongoClient("mongodb+srv://SCP:PASSWORDHAHA@scp16cluseter.foubt.mongodb.net/myFirstDatabase?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE")


def getDashboardURL():
    return "http://scp16tsundere.pagekite.me:443"

class botUser(object):
    def __init__(self, user:discord.Member):
        self.user = user
        self.inv = mulah.find_one({"id":user.id}, {"inv"})["inv"]
        self.gf = mulah.find_one({"id":user.id}, {"gf"})["gf"]
        self.lp = mulah.find_one({"id":user.id}, {"lp"})["lp"]
        self.breakups = mulah.find_one({"id":user.id}, {"breakups"})["breakups"]
        self.kisses = mulah.find_one({"id":user.id}, {"kisses"})["kisses"]
        self.boinks = mulah.find_one({"id":user.id}, {"boinks"})["boinks"]
        self.money = mulah.find_one({"id":user.id}, {"money"})["money"]
        self.job = mulah.find_one({"id":user.id}, {"job"})["job"]
        self.duelwins = mulah.find_one({"id":user.id}, {"duelwins"})["duelwins"]
        self.duelloses = mulah.find_one({"id":user.id}, {"duelloses"})["duelloses"]
        self.duelretreats = mulah.find_one({"id":user.id}, {"duelretreats"})["duelretreats"]
        self.watchlist = mulah.find_one({"id":user.id}, {"watchlist"})["watchlist"]
        self.achievements = mulah.find_one({"id":user.id}, {"achievements"})["achievements"]
        self.proposes = mulah.find_one({"id":user.id}, {"proposes"})["proposes"]
        self.dates = mulah.find_one({"id":user.id}, {"dates"})["dates"]
        self.relationships = mulah.find_one({"id":user.id}, {"relationships"})["relationships"]
        self.gambles = mulah.find_one({"id":user.id}, {"gambles"})["gambles"]
        self.gamblewins = mulah.find_one({"id":user.id}, {"gamblewins"})["gamblewins"]
        self.upgradepoints = mulah.find_one({"id":user.id}, {"upgradepoints"})["upgradepoints"]
        self.gameskill = mulah.find_one({"id":user.id}, {"gameskill"})["gameskill"]
        self.bank = mulah.find_one({"id":user.id}, {"bank"})["bank"]
        self.net = mulah.find_one({"id":user.id}, {"net"})["net"]
        self.abilityxp = mulah.find_one({"id":user.id}, {"abilityxp"})["abilityxp"]
        self.mmorpg = mulah.find_one({"id":user.id}, {"mmorpg"})["mmorpg"]

    def updateWholeMongo(self):
        dictionaryOfAttributes = self.__dict__
        for x in dictionaryOfAttributes:
            mulah.update_one({"id", self.user.id}, {"$set":{x:dictionaryOfAttributes[x]}})
    
    def updateOne(self, attribute):
        attribute = attribute.lower()
        dictionaryOfAttributes = self.__dict__
        try:
            mulah.update_one({"id":self.user.id}, {"$set":{attribute:dictionaryOfAttributes[attribute]}})
        except:
            pass

    def incOne(self, attribute, value:int):
        attribute = attribute.lower()
        try:
            mulah.update_one({"id":self.user.id}, {"$inc":{attribute:value}})
        except:
            pass





cluster = getMongo()
mulah = cluster["discord"]["mulah"]
levelling = cluster["discord"]["levelling"]
DiscordGuild = cluster["discord"]["guilds"]
achievements = [
    {"name":"First Kiss!", "desc":"Kiss someone for the first time!", "category":"relationships"},
    {"name":"Virginity Loss!", "desc":"Boink someone for the first time!", "category":"relationships"},
    {"name":"Engaged!", "desc":"Propose to someone for the first time!", "category":"relationships"},
    {"name":"Jerk", "desc":"Turns out you were the problem", "category":"relationships"},
    {"name":"Divorcee!", "desc":"Get a life bro.", "category":"relationships"},
    {"name":"First Date!", "desc":"First date with GF!", "category":"relationships"},
    {"name":"bobs", "desc":";)", "category":"relationships"},
    

    {"name":"Getting By", "desc":"finally making some money! good job!", "category":"finance"},
    {"name":"Millionaire!", "desc":"its what it sounds like", "category":"finance"},
    
    {"name":"Billionaire!", "desc":"Treat your workers with respect.", "category":"finance"},
    {"name":"Employed!", "desc":"You got a job.", "category":"finance"},
    {"name":"Gambler!", "desc":"You gambled for the first time! ", "category":"finance"},
    {"name":"Winner!", "desc":"You won a gamble! ", "category":"finance"},


    {"name":"Death!", "desc":"Get a life bro.", "category":"gaming"},
    {"name":"virgin", "desc":"Secret!", "category":"gaming"},
    {"name":"FloorGang", "desc":"Secret!", "category":"gaming"},
    {"name":"First PC!", "desc":"Create your first PC!", "category":"gaming"},
    {"name":"Linus Tech Tips", "desc":"Create a beefy Computer with at least 12000 power!", "category":"gaming"},
    {"name":"True Gamer", "desc":"Install 5 games on a single PC!", "category":"gaming"},

    
]



def gamble(odds:int, times:int):
    count = 0
    wins = 0
    while count<=times:
        number = random.randint(1,odds)
        if number == 1:
            wins+=1
        else:
            pass
        count+=1
    return wins
    




def GetFirstKey(dict:dict):
    for x in dict:
        return x


def removeDupes(test_list:list):
    res =[]
    for i in test_list:
        if i not in res:
            res.append(i)
    return res


class chat(object):
    def __init__(self, chatlog):
        self.chatlog = chatlog
    
    def ask(self,question):
        response = openai.Completion.create(
            engine="davinci",
            prompt=self.chatlog + question + "\nAI:",
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=["\n", " Human:", " AI:"]
        )
        return response["choices"][0]["text"]
    
    def append_interaction_to_chat_log(self, answer):
        self.chatlog += "AI:" +answer+"\n"
    



#openai api completions.create -m ada:ft-sentientproductions-2021-12-27-00-47-10 -p "*bad text*"


##-------------------------------------------------------------INV FUNCTS
def RemoveFromInventory(user, item, AmountToRemove:int=None):
    if AmountToRemove==None:
        AmountToRemove=1
    inv = mulah.find_one({"id":user.id}, {"inv"})["inv"]
    itemdict = next(x for x in inv if x["name"].lower() ==item.lower())
    itemdict["amount"]-=AmountToRemove
    if itemdict["amount"]==0:
        inv.remove(itemdict)
    mulah.update_one({"id":user.id}, {"$set":{"inv":inv}})

def AddToInventory(user, item, ReferenceList:list, AmountToAdd:int=None):
    if AmountToAdd==None:
        AmountToAdd=1
    inv = mulah.find_one({"id":user.id}, {"inv"})["inv"]
    itemdict = next((x for x in inv if x["name"].lower() ==item.lower()), None)
    ThingToAdd = next(x for x in ReferenceList if x["name"].lower()==item.lower())
    if itemdict != None:
        itemdict["amount"]+=AmountToAdd
    else:
        inv.append({"name":ThingToAdd["name"], "amount":AmountToAdd, "desc": "%s"%(ThingToAdd["desc"])})
    mulah.update_one({"id":user.id}, {"$set":{"inv":inv}})


def InvCheck(user, item:str, Id=False, amount:int=1) -> bool:
    if Id==False:
        inv = mulah.find_one({"id":user.id}, {"inv"})["inv"]
        check = next((x for x in inv if x["name"].lower()==item.lower() and x["amount"]>=amount), None)
        if check == None:
            return False
        else:
            return True
    else:
        inv = mulah.find_one({"id":user}, {"inv"})["inv"]
        check = next((x for x in inv if x["name"].lower()==item.lower() and x["amount"]>=amount), None)
        if check == None:
            return False
        else:
            return True


def InvCheckWithItem(user, item:str, Id=False, amount:int=1):
    if Id==False:
        user = user.id
    inv = mulah.find_one({"id":user}, {"inv"})["inv"]
    check = next((x for x in inv if x["name"].lower()==item.lower() and x["amount"]>=amount and "parts" not in x.keys()), None)
    if check == None:
        return False
    else:
        return check
















##----------------------------------------------------Achievement Functs
def XpBar(val, max, fill=":blue_square:", empty=":white_large_square:", NumOfSquares=20, righttoleft=False):
    if righttoleft:
        valueOfBlue = math.floor((val/max)*NumOfSquares)
        if valueOfBlue<0:
            return empty*NumOfSquares
        valueofWhite = NumOfSquares-valueOfBlue
        finalstr = empty*valueofWhite+fill*valueOfBlue
        return finalstr   
    else:
        valueOfBlue = math.floor((val/max)*NumOfSquares)
        if valueOfBlue<0:
            return empty*NumOfSquares
        valueofWhite = NumOfSquares-valueOfBlue
        finalstr = fill*valueOfBlue+empty*valueofWhite
        return finalstr
    
    
def GetKeysFromDictInList(list:list):
    keys= []
    for x in list:
        for z in x.keys():
            keys.append(z)
    return keys

def GetLevel(id):
    xp = levelling.find_one({"id":id}, {"xp"})["xp"]
    lvl = 0
    while True:
        if xp < ((50*(lvl**2))+(50*(lvl))):
            break
        lvl+=1
    return lvl

def getLevelfromxp(xp):
    lvl = 0
    while True:
        if xp < ((50*(lvl**2))+(50*(lvl))):
            break
        lvl+=1
    return lvl

def achievementcheck(user,achievement:str):
    try:
        value = mulah.find_one({"id":user.id}, {"achievements"})["achievements"]
        if achievement in value:
            return "✅"
        else:
            return "❌"
    except:
        return "❌"
        
def achievementpercent(achievement:str):
    count = 0
    achCount=0
    for x in mulah.find():
        count+=1
        try:
            if achievement in x["achievements"]:
                achCount+=1
        except:
            pass
    return (achCount/count)*100

def ChoiceParts(choices:list, ReactionsList = ['1️⃣', '2️⃣', '3️⃣', '4️⃣','5️⃣','6️⃣','7️⃣','8️⃣','9️⃣','🔟']):
    count = 0
    reactionlist = []
    emptydict = {}
    finalstr = ""
    for x in choices:
        emptydict[ReactionsList[count]]=x
        reactionlist.append(ReactionsList[count])
        finalstr+="%s %s\n"%(ReactionsList[count], x)
        count+=1
    return [emptydict, finalstr, reactionlist]


async def AchievementEmbed(ctx, EarnedAchievement):
    yourachievements = mulah.find_one({"id":ctx.author.id}, {"achievements"})["achievements"]
    AchievementDict = next(x for x in achievements if x["name"]==EarnedAchievement)
    if AchievementDict["name"] not in yourachievements:
        print(AchievementDict["name"])
        embed = discord.Embed(title = "Congratulations! you earned the achievement %s"%(AchievementDict["name"]), description = AchievementDict["desc"], color = discord.Color.gold())
        embed.set_image(url = 'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/socialmedia/apple/271/trophy_1f3c6.png')
        yourachievements.append(AchievementDict["name"])
        mulah.update_one({"id":ctx.author.id}, {"$set":{"achievements":yourachievements}})
        embed.set_author(name = ctx.author.display_name, icon_url=ctx.author.avatar_url)
        await ctx.channel.send(embed=embed)    





##-------------------------------------------------------------------------GLOBAL VARIABLES, DATASETS

def getEmotionList():
    return ["embarrassed", "horny","surprised","climax", "image", "bed", "angry", "fear", "sad", "dissapointed"]

def getBegList():
    return {
        1:{"name":"Jake Paul", "value":"ew get away from me", "value":2},
        2:{"name":"Mrbeast", "value":"Oh heres 10 grand without the grand", "amount":10},
        3:{"name":"Joe Biden", "value":"u smell nice today", "amount":9},
        4:{"name":"Naruto", "value":"hiruzen gave me less","amount":1},
        5:{"name":"Luffy", "value":"have some bro","amount":5},
        6:{"name":"Alien", "value":"Damn capitalism sucks","amount":11},
        7:{"name":"The Rock", "value":"passive income baby woo", "amount":6},
        8:{"name":"zendaya", "value":"idk what zendaya says bruh", "amount":8},
        9:{"name":"Louis XVI", "value":"hey man have some bread", "amount":19},
        10:{"name":"Askeladd", "value":"have some gold", "amount":10},

    }

def getAchievementList():
    return achievements

def getWorkLists():
    return [
            {"name":"McDonalds worker", "salary":15, "req":1, "words":["bigmac", "burger", "broken"], "sentences":["sorry, the icecream machine is broken", "what can I get for you?", "welcome to mcdonalds"]},
            {"name":"Gamer", "salary":150, "req": 5, "words":["dorito", "mechanical", "virgin"], "sentences":["i hate lag", "hes one tap", "what a sweat"]},
            {"name":"Bitcoin Miner", "salary":250, "req": 10, "words":["nvidia", "shortage", "shameless"], "sentences":["People hate me for a good reason", "that is passive income", "I like cheese"]},
            {"name":"Youtuber", "salary":450, "req": 15, "words":["subscribe", "like", "rich"], "sentences":["make sure to smash that like button", "i dont know how to start this video", "leave your memes in the subreddit"]},
            {"name":"Business Man", "salary":160, "req":20, "words":["business", "passive", "pigeon"], "sentences":["sorry thats not passive income", "it is ten times cheaper to keep a customer than to get a new one"]},
            {"name":"Jeff bezos", "salary":1000000000, "req":100, "words":["abuse", "rocket", "money"], "sentences":["I love money", "I appreciate the lower class", "i am not a supervillain"]},
        ]

def getShopItems():
    return [
            {"name":"phone", "value":800, "desc":"Text your Girlfriend!"},
            {"name": "netflixsub", "value": 29, "desc": "Netflix and chill with your gf"},
            {"name": "lotteryticket", "value": 2, "desc": "A chance to win 1 million dollars"},
            {"name": "movieticket", "value" : 16, "desc":"watch a movie with your gf"},
            {"name": "ring", "value" : 10000, "desc":"propose to your gf"},

        ]

def getBattleItems():
    return [
            {"name":"UpgradePoint", "value":2000, "desc":"`^upgrade` one of your stats!"},

            {"name":"Vaccine", 
            "type":"Heal",
            "desc":"Heal ig", 
            "rarity":"Legendary",
            "value":2000,
            "abilities":{"vaccine":1}},

            {"name":"Saitamas Dish Gloves", 
            "type":"hands", 
            "desc":"The Most powerful item in the game.",
            "rarity":"illegal", 
            "value":1000000000,
            "attribute":{"strength":1000000}},

            {"name":"Sharingan", 
            "type":"head", 
            "desc":"Op doujutsu",
            "rarity":"Legendary", 
            "value":200000,
            "abilities":{"Amaterasu":1, "Susanoo":1}},


            {"name":"Demon Destroyer", 
            "type":"primary", 
            "desc":"Can deflect spells completely!", 
            "rarity":"Legendary", 
            "value":20000,
            "abilities":{"Black Slash":1, "Deflect":1, "Black Divider":1}
            },

            {"name":"Sword", 
            "type":"primary", 
            "desc":"Basic sword.", 
            "rarity":"Common", 
            "value":200,
            "abilities":{"Slash":1}
            },

            {"name":"Spear", 
            "type":"primary", 
            "desc":"Basic weapon.", 
            "rarity":"Common", 
            "value":200,
            "abilities":{"Pierce":1}
            },

        ]

def getToolValues():
    return [
            {"name": "rifle", "value" : 400, "desc":"`^hunt` to get animals!"},
            {"name": "fishpole", "value" : 100, "desc":"`^fish` to catch fish!"},
            {"name":"pickaxe", "durability":59, "fortune":1, "craft":{"wood":5}, "value":25, "desc":"cheap mining"},
            {"name":"iron pickaxe", "durability":250, "fortune":2, "craft":{"wood":2, "iron":3}, "value":25, "desc":"better mining"},
            {"name":"gold pickaxe", "durability":33, "fortune":4, "craft":{"wood":2, "gold":3}, "value":115, "desc":"fine mining"},
            {"name":"diamond pickaxe", "durability":1562, "fortune":4, "craft":{"wood":2, "diamond":3}, "value":13010, "desc":"best mining"},

            {"name":"axe", "durability":59, "fortune":1, "craft":{"wood":4}, "value":29, "desc":"Chop wood"},
            {"name":"iron axe", "durability":250, "fortune":2, "craft":{"wood":2, "iron":3}, "value":25, "desc":"Chop more wood"},
            {"name":"gold axe", "durability":33, "fortune":4, "craft":{"wood":2, "gold":3}, "value":115, "desc":"Chop lots of wood"},
            {"name":"diamond axe", "durability":1562, "fortune":4, "craft":{"wood":2, "diamond":3}, "value":13010, "desc":"Chop even more wood"},

            {"name":"hoe", "durability":59, "fortune":1, "craft":{"wood":2}, "value":10, "desc":"Farm stuff idk"},
            {"name":"iron hoe", "durability":250, "fortune":2, "craft":{"wood":2, "iron":2}, "value":20, "desc":"Farm stuff idk"},
            {"name":"gold hoe", "durability":32, "fortune":4, "craft":{"wood":2, "gold":2}, "value":80, "desc":"Farm stuff idk"},
            {"name":"diamond hoe", "durability":1561, "fortune":4, "craft":{"wood":2, "diamond":2}, "value":8810, "desc":"Farm stuff idk"},
        ]

def getFarmItems():
    return [
            {"name":"uncommon fish", "value":10, "desc":"cheap fish to sell"},
            {"name":"common fish", "value":20, "desc":"a mediocre fish"},
            {"name":"rare fish", "value":50, "desc":"high quality fish"},
            {"name":"legendary fish", "value":150, "desc":"very valuable fish"},
            {"name":"mouse", "value":10, "desc":"idk why someone would even bother"},
            {"name":"rabbit", "value":50, "desc":"tste great in stew"},
            {"name":"deer", "value":150, "desc":"sells well"},
            {"name":"bigfoot", "value":1000, "desc":"make some mulah"},
            {"name":"coal", "value":1, "desc":"non renewable energy source"},
            {"name":"iron", "value":5, "desc":"for what"},
            {"name":"gold", "value":35, "desc":"terrible durability"},
            {"name":"diamond", "value":4400, "desc":"sells for a lot"},
            {"name":"ruby", "value":10000, "desc":"One of the most precious things in this world"},
            {"name":"wheat", "value":10, "desc":"carbs"},
            {"name":"beetroot", "value":20, "desc":"why do people eat this"},
            {"name":"melon", "value":50, "desc":"mmm"},
            {"name":"pumpkin", "value":150, "desc":"pumpkin pie tastes great"},
            {"name":"wood", "value":5, "desc":"profits pile up"},

        ]
def getPcItems():
    return [
            {"name":"4gbRam", "type":"ram", "value": 20,"desc":"Use this for your PC!","power":0,"space":0, "rspace": 4000, "synthesis":0, "consumption":10},
            {"name":"8gbRam", "type":"ram", "value": 50, "desc":"Reasonable upgrade!","power":0,"space":0, "rspace": 8000, "synthesis":0, "consumption":10},
            {"name":"16gbRam", "type":"ram", "value": 100, "desc":"Do you really need this?","power":0,"space":0, "rspace": 16000, "synthesis":0, "consumption":10},
            {"name":"32gbRam", "type":"ram", "value": 200, "desc":"Thats overkill man, but you do you ig.","space":0,"power":0, "rspace": 32000, "synthesis":0, "consumption":10},
            {"name":"i5","type":"cpu", "value": 160, "desc":"A perfect cpu- if you are on a budget","space":0,"rspace":0, "power":1500 , "synthesis":0, "consumption":250},
            {"name":"i7","type":"cpu", "value": 250, "desc":"Great for upper middle range machines!","space":0, "power":2000,"rspace":0, "synthesis":0, "consumption":250 },
            {"name":"i9","type":"cpu", "value": 370, "desc":"A great gaming cpu overall.","space":0, "power":2500,"rspace":0, "synthesis":0, "consumption":250 },
            {"name":"threadripper","type":"cpu", "value": 3000, "desc":"An excellent cpu that will never know pain.","space":0, "power":4000,"rspace":0, "synthesis":0, "consumption":280 },
            {"name":"xeon","type":"cpu", "value": 10000, "desc":"For NASA computers", "power":10000,"space":0,"rspace":0, "synthesis":0, "consumption":350},
            {"name":"512SSD","type":"storage", "value": 70, "desc":"Great storage for a decent machine!","rspace":0,"power":0, "synthesis":0, "space": 512000, "consumption":10},
            {"name":"1TBSSD","type":"storage", "value": 100, "desc":"This should be enough for most people","rspace":0,"power":0, "synthesis":0, "space": 1000000, "consumption":10 },
            {"name":"4TBSSD","type":"storage", "value": 500, "desc":"enough storage for your homework folder","rspace":0,"power":0, "synthesis":0, "space": 4000000, "consumption":10 },
            {"name":"1660ti","type":"gpu", "value": 280, "desc":"entry level gpu","space":0, "power":1500,"rspace":0, "synthesis":0,"consumption":120  },
            {"name":"1080ti","type":"gpu", "value": 1074, "desc":"Good for mid range machines","space":0, "power":2000,"rspace":0, "synthesis":0, "consumption":250 },
            {"name":"2080ti","type":"gpu", "value": 1376, "desc":"imagine using a 20 series","space":0, "power":2500,"rspace":0, "synthesis":0, "consumption":275 },
            {"name":"3080ti","type":"gpu", "value": 3000, "desc":"Scalper price!", "space":0, "power":6000,"rspace":0, "synthesis":0, "consumption":350 },
            {"name":"650watt","type":"psu", "value": 5000, "desc":"scalper price!","space":0,"power":0, "synthesis":650,"rspace":0, "consumption":0  },
            {"name":"750watt","type":"psu", "value": 5000, "desc":"scalper price!","space":0,"power":0, "synthesis":750,"rspace":0, "consumption":0  },
            {"name":"850watt","type":"psu", "value": 5000, "desc":"scalper price!","space":0,"power":0, "synthesis":850,"rspace":0, "consumption":0  },
            {"name":"900watt","type":"psu", "value": 5000, "desc":"scalper price!","space":0,"power":0, "synthesis":900,"rspace":0, "consumption":0  },
            {"name":"motherboard","type":"board", "value": 100, "desc":"build a pc.","space":0,"power":0, "synthesis":0,"rspace":0, "consumption":0  }


        ]  

def getGameItems():
    return [
            {"name":"Minecraft", "genre":["adventure", "creativity"],"space":1500, "value":26, "desc": "anything can run Minecraft!", "lpincrease":30, "recommendedspecs":{"totalram":8000, "power":1500}},
            {"name":"Fortnite", "genre":["fps"],"space":49000, "value":0, "desc": "How much lp were you expecting for fortnite?", "lpincrease":5, "recommendedspecs":{"totalram":8000, "power":2500}},
            {"name":"Valorant", "genre":["fps"],"space":14400, "value":0, "desc": "spend 80% of the game spectating.", "lpincrease":25, "recommendedspecs":{"totalram":8000, "power":3000}},
            {"name":"Terraria", "genre":["adventure", "creativity"],"space":100, "value":5, "desc": "A great friend of Mc", "lpincrease":20, "recommendedspecs":{"totalram":8000, "power":1500}},
            {"name":"Microsoft Flight simulator", "genre":["creativity"],"space":150000, "value":60, "desc": "You probably cant run this.", "lpincrease":40, "recommendedspecs":{"totalram":16000, "power":5000}},
            {"name":"Crysis 3", "genre":["adventure"],"space":17000, "value":5, "desc": "Your pc simply cant run this.", "lpincrease":50, "recommendedspecs":{"totalram":32000, "power":7800}},
            {"name":"League of Legends", "genre":["strategy"],"space":22000, "value":0, "desc": "Dont do it.", "lpincrease":-50, "recommendedspecs":{"totalram":8000, "power":2800}}
        ]

def getGameWords():
    return [
            {"name": "Minecraft", "words":["block", "redstone", "blockhit", "endcrystal"]},
            {"name": "Fortnite", "words":["build", "ninja", "virgin", "clap"]},
            {"name": "Valorant", "words":["hipfire", "slow", "spectator", "Operator"]},
            {"name": "Terraria", "words":["Terraria", "cheap", "fun", "pewdiepie"]},
            {"name": "Microsoft Flight Simulator", "words":["plane", "aviation", "pilot", "graphics"]},
            {"name": "Crysis 3", "words":["Block", "redstone", "blockhit", "endcrystal"]},
            {"name": "League of Legends", "words":["virgin", "discordmod", "glasses", "asian"]},
        ]

def getEnemyList():
    return [
            {"name":"Acnologia", 
            "health":5000, 
            "strength":800, 
            "defense":400, 
            "intelligence":1000,
            "mana":1000,
            "image":"https://static.wikia.nocookie.net/vsbattles/images/7/71/New_Human_Acnologia_Render.png/revision/latest/scale-to-width-down/400?cb=20200704092623", 
            "size":((160, 199)), 
            "paste":((468,125)),
            "abilities":{"Fire Ball":1,"Absorb":1,"vaccine":1}
            }
        ]

def getClassDict():
    return [
            {"class":"warrior", 
            "desc":"Warrior class. Great all around class.", 
            "stats":{"strength":50, "defense":50, "intelligence":30, "sense":20, "health":100, "CurrentHealth":100}, 
            "ability":"Rage", 
            "abilitydesc":"Increase attack damage by 50%"},

            {"class":"assassin", 
            "desc":"Assassin class. deadly damage output, low defense.", 
            "stats":{"strength":110, "defense":15, "intelligence":30, "sense":50, "health":80, "CurrentHealth":100}, 
            "ability":"stealth", 
            "abilitydesc":"Become invisible! All attacks will deal full damage, ignoring opponents' defense stat."},

            {"class":"Mage", 
            "desc":"Mage class. Uses movie science", 
            "stats":{"strength":40, "defense":30, "intelligence":100, "sense":60, "health":100, "CurrentHealth":100}, 
            "ability":"Fire ball", 
            "abilitydesc":"Send a fire ball at your enemies!"},

            {"class":"Healer", 
            "desc":"Healer class. Can heal. A lot.", 
            "stats":{"strength":40, "defense":50, "intelligence":80, "sense":30, "health":150, "CurrentHealth":150}, 
            "ability":"Heal!", 
            "abilitydesc":"50% HP boost!"}

        ]

def getEffectDict():
    return [
            {"name":"Bleed","type":"Physical", "category":["health"], "AffectsSender":False, "value":95, "length":4, "ValSet":False},
            {"name":"Defenseless","type":"Physical", "category":["defense"], "AffectsSender":False, "value":10, "length":3, "ValSet":True},
            {"name":"Regeneration","type":"Physical", "category":["health"], "AffectsSender":False, "value":115, "length":4, "ValSet":True},
            {"name":"Amaterasu","type":"Magic", "category":["health"], "AffectsSender":False, "value":80, "length":1000, "ValSet":False},
            {"name":"Susanoo","type":"Magic", "category":["defense"], "AffectsSender":True, "value":1000, "length":1000, "ValSet":False},


        ]
def getBackgroundList():
    return [
            {"name":"house.jpg", "paste":(378,167), "size":(377,467)},
            {"name":"nightsky.jpg", "paste":(60,82), "size":(195,279)},
            {"name":"macd.jpg", "paste":(72,6), "size":(204,310)},
            {"name":"OliveGarden.jpg", "paste":(133,155), "size":(203,310)},
            {"name":"redlobster.jpg", "paste":(213,77), "size":(191,254)}
        ]



def getRestaurants():
    return [
            {
            "name":"olive garden", 
            "menu":{
                "Chicken and Shrimp Carbonara":27, 
                "Chicken Parmigiana":23, 
                "Shrimp Alfredo":26, 
                "Chicken Marsala":24, 
                "Cheese Ravioli":19,
                "Herb Grilled Salmon":29,
                "6oz sirloin":25
                }, 
            "background":"OliveGarden.jpg", 
            "img":"image", 
            "waiter":"OliveGardenWaiter.jpg"
            },

            {
            "name":"Red Lobster",
            "menu":{
                "wild caught flounder":10,
                "Shrimp Linguini Alfredo":11,
                "Lobster pizza":11,
                "clam chowder":5,
                "classic caesar salad":10

            },
            "background":"redlobster.jpg",
            "img":"image",
            "waiter":"RedLobsterWaiter.jpg"

            },

            {
            "name":"mcdonalds", 
            "menu":{
                "bigmac":6, 
                "QuarterPounder":3, 
                "Bacon Clubhouse Burger":4, 
                "fillet-o-fish":3, 
                "happy meal":4
                }, 
            "background":"macd.jpg", 
            "img":"image", 
            "waiter":"McdonaldsWaiter.jpg"
            }
        ]

def getDateTalkMap():
    return [
            {"typename":"Tsundere", "invite":["{author}! Im hungry! lets go eat!"], "react":["hmm, lets eat at {restaurant}!"], "whattoeat":["hmm, I'll order the {order}!, what will you have, {author}?"]},
            {"typename":"Dandere", "invite":["{author}! Im hungry! lets go eat!"], "react":["hmm, lets eat at {restaurant}!"], "whattoeat":["hmm, I'll order the {order}!, what will you have, {author}?"]},
            {"typename":"Kuudere", "invite":["{author}! Im hungry! lets go eat!"], "react":["hmm, lets eat at {restaurant}!"], "whattoeat":["hmm, I'll order the {order}!, what will you have, {author}?"]},
            {"typename":"Sadodere", "invite":["{author}! Im hungry! lets go eat!"], "react":["hmm, lets eat at {restaurant}!"], "whattoeat":["hmm, I'll order the {order}!, what will you have, {author}?"]},
            {"typename":"Kamidere", "invite":["{author}! Im hungry! lets go eat!"], "react":["hmm, lets eat at {restaurant}!"], "whattoeat":["hmm, I'll order the {order}!, what will you have, {author}?"]},
            {"typename":"Sweet", "invite":["{author}! Im hungry! lets go eat!"], "react":["hmm, lets eat at {restaurant}!"], "whattoeat":["hmm, I'll order the {order}!, what will you have, darling?"]},
            {"typename":"Yandere", "invite":["{author}! Im hungry! lets go eat!"], "react":["hmm, lets eat at {restaurant}!"], "whattoeat":["hmm, I'll order the {order}!, what will you have, {author}?"]},
        ]


def getTalkMap():
    return [
            [
            {"typename":"Sweet", "action":"none","img":"image", "response":"see you, {0}! I love you!", "background":"house.jpg"},
            {"map":["sad", "scared", "happy", "angry", "horny"], "action":"Im not feeling that way","img":"image", "response":["Im sorry, how are you feeling right now?"], "background":"house.jpg"},
            {"map":["accept invitation"], "action":"horny","img":"embarrassed", "response":["ohh, thats what you were feeling. Thats ok, I can help you out with that ;)"], "background":"house.jpg"},
            {"map":["leave","invite her to go do something"], "action":"end","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["No, Im fine", "Im not feeling that way"], "action":"sad","img":"image", "response":["It seems like you are sad. is that right? Thats too bad! is there anything I can do?"], "background":"house.jpg"},
            {"map":["lie on lap","dont lie on lap"], "action":"No, Im fine","img":"image", "response":["I cant give much, but I will support you will all Ive got! come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["leave","lie on lap"], "action":"dont lie on lap","img":"image", "response":["come on, just for a little while? come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["end"], "action":"lie on lap","img":"image","response": ["Hey. I know you can do it. I love you so much. That will never change. \n*You are my sunshine, My only sunshine\n You make me happy when skies are gray!\n You'll never know, dear, how much I love you!\n please dont take my sunshine away!*\n Did you like my voice? I hope so! *smooch*"], "background":"house.jpg"},
            {"map":["hug","kiss","Im really thankful for you!", "Im not feeling that way"], "action":"happy","img":"image", "response":["Thats amazing! im so happy for you!"], "background":"house.jpg"},
            {"map":["hug", "kiss", "end"], "action":"Im really thankful for you!","img":"image", "response":["aww, I love you so much! of course Id support you!"], "background":"house.jpg"},
            {"map":["end","kiss","invite her to go do something"], "action":"hug","img":"image", "response":["hmmm? You want a hug? of course!! *sqeezes*"], "background":"house.jpg"},
            {"map":["end","invite her to go do something"], "action":"kiss","img":"image", "response":["*mwah* I'll see you around! I love you!"], "background":"house.jpg"},
            {"map":["kiss"], "action":"Im really thankful for you!","img":"image", "response":["hey. I care about you!! Its only normal.."], "background":"house.jpg"},
            {"map":["walk away","lie on lap", "Im not feeling that way"], "action":"angry","img":"image", "response":["Hey. Im not sure if you are in the mood, you seem mad, or annoyed, but wanna rest on my lap?"], "background":"house.jpg"},
            {"map":["go with her", "dont follow"], "action":"walk away","img":"image", "response":["Hey I know just the thing! Follow me!"], "background":"house.jpg"},
            {"map":["end"], "action":"dont follow","img":"image", "response":["all right. I understand, Ill give you some time. If you wanna talk to me about anything, Im always available to you!"], "background":"house.jpg"},
            {"map":["leave","look at stars"], "action":"go with her","img":"image", "response":["this is the night sky! It looks nice, right? You can relax here. I find it nice gazing at the stars"], "background":"nightsky.jpg"},
            {"map":["end"], "action":"leave","img":"image", "response":["Im always ready to talk if you need me. I love you! bye!"], "background":"house.jpg"},
            {"map":["end"], "action":"look at stars","img":"image", "response":["Its nice right? Ill leave you be for now."], "background":"nightsky.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scared","img":"image", "response":["It sounds like you are scared. I know you are strong. You are also smart! if you cant handle it on your own, find someone to help you! You shouldnt always try to do things on your own!"], "background":"house.jpg"},
        
            {"map":["comfortgf"], "action":"sadgf","img":"sad", "response":["{0}, im feeling really sad! I dont like this! do something!"], "background":"house.jpg"},
            {"map":["gaming", "movies", "netflix", "horny"], "action":"invite her to go do something","img":"angry", "response":["What do you want to do together? Im... im open to anything!"], "background":"house.jpg"},            
            {"map":["accept invitation"], "action":"comfortgf1","img":"embarrassed", "response":["{0}! I.. I wanna f***. Im feeling horny af rn."], "background":"house.jpg"},
            {"map":["gaming"], "action":"comfortgf2","img":"image", "response":["I.. I wanna game."], "background":"house.jpg"},
            {"map":["netflix"], "action":"comfortgf3","img":"image", "response":["I.. I wanna watch netflix. "], "background":"house.jpg"},
            {"map":["movies"], "action":"comfortgf4","img":"image", "response":["I.. I wanna watch a movie."], "background":"house.jpg"},
            {"map":["comfortgf1", "comfortgf2", "comfortgf3","comfortgf4"], "action":"comfortgf","img":"angry", "response":["Hmm, I think I know what will cheer me up!"], "background":"house.jpg"},
            {"map":["great", "Im not feeling that way"], "action":"happygf","img":"image", "response":["Hey! How are you doing?"], "background":"house.jpg"},
            {"map":["yeah sure what?"], "action":"great","img":"embarrassed", "response":["I wanna go do something with you..."], "background":"house.jpg"},
            {"map":["gaming"], "action":"gaming","img":"image", "response":["{0}! Lets play a game! I havnt played with you in forever!!"], "background":"house.jpg"},
            {"map":["movies"], "action":"movies","img":"image", "response":["{0}! Lets watch a movie!"], "background":"house.jpg"},
            {"map":["netflix"], "action":"netflix","img":"image", "response":["{0}! Lets watch netflix"], "background":"house.jpg"},
            {"map":["kiss","hug"], "action":"attentionwant","img":"embarrassed", "response":["I want attention!"], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scaredgf","img":"angry", "response":["Im not sure how im gonna pay the rent."], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"angrygf","img":"angry", "response":["Im not having a good day. I just wanna go to sleep."], "background":"house.jpg"},
        ],


            [
            {"typename":"Tsundere", "action":"none","img":"image", "response":"see you, {0}! I love you!", "background":"house.jpg"},
            {"map":["sad", "scared", "happy", "angry", "horny"], "action":"Im not feeling that way","img":"image", "response":["Im sorry, how are you feeling right now?"], "background":"house.jpg"},
            {"map":["accept invitation"], "action":"horny","img":"embarrassed", "response":["ohh, thats what you were feeling. Thats ok, I can help you out with that ;)"], "background":"house.jpg"},
            {"map":["leave","invite her to go do something"], "action":"end","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["No, Im fine", "Im not feeling that way"], "action":"sad","img":"image", "response":["It seems like you are sad. is that right? Thats too bad! is there anything I can do?"], "background":"house.jpg"},
            {"map":["lie on lap","dont lie on lap"], "action":"No, Im fine","img":"image", "response":["I cant give much, but I will support you will all Ive got! come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["leave","lie on lap"], "action":"dont lie on lap","img":"image", "response":["come on, just for a little while? come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["end"], "action":"lie on lap","img":"image","response": ["Hey. I know you can do it. I love you so much. That will never change. \n*You are my sunshine, My only sunshine\n You make me happy when skies are gray!\n You'll never know, dear, how much I love you!\n please dont take my sunshine away!*\n Did you like my voice? I hope so! *smooch*"], "background":"house.jpg"},
            {"map":["hug","kiss","Im really thankful for you!", "Im not feeling that way"], "action":"happy","img":"image", "response":["Thats amazing! im so happy for you!"], "background":"house.jpg"},
            {"map":["hug", "kiss", "end"], "action":"Im really thankful for you!","img":"image", "response":["aww, I love you so much! of course Id support you!"], "background":"house.jpg"},
            {"map":["end","kiss","invite her to go do something"], "action":"hug","img":"image", "response":["hmmm? You want a hug? of course!! *sqeezes*"], "background":"house.jpg"},
            {"map":["end","invite her to go do something"], "action":"kiss","img":"image", "response":["*mwah* I'll see you around! I love you!"], "background":"house.jpg"},
            {"map":["kiss"], "action":"Im really thankful for you!","img":"image", "response":["hey. I care about you!! Its only normal.."], "background":"house.jpg"},
            {"map":["walk away","lie on lap", "Im not feeling that way"], "action":"angry","img":"image", "response":["Hey. Im not sure if you are in the mood, you seem mad, or annoyed, but wanna rest on my lap?"], "background":"house.jpg"},
            {"map":["go with her", "dont follow"], "action":"walk away","img":"image", "response":["Hey I know just the thing! Follow me!"], "background":"house.jpg"},
            {"map":["end"], "action":"dont follow","img":"image", "response":["all right. I understand, Ill give you some time. If you wanna talk to me about anything, Im always available to you!"], "background":"house.jpg"},
            {"map":["leave","look at stars"], "action":"go with her","img":"image", "response":["this is the night sky! It looks nice, right? You can relax here. I find it nice gazing at the stars"], "background":"nightsky.jpg"},
            {"map":["end"], "action":"leave","img":"image", "response":["Im always ready to talk if you need me. I love you! bye!"], "background":"house.jpg"},
            {"map":["end"], "action":"look at stars","img":"image", "response":["Its nice right? Ill leave you be for now."], "background":"nightsky.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scared","img":"image", "response":["It sounds like you are scared. I know you are strong. You are also smart! if you cant handle it on your own, find someone to help you! You shouldnt always try to do things on your own!"], "background":"house.jpg"},
        
            {"map":["comfortgf"], "action":"sadgf","img":"sad", "response":["{0}, im feeling really sad! I dont like this! do something!"], "background":"house.jpg"},
            {"map":["gaming", "movies", "netflix", "horny"], "action":"invite her to go do something","img":"angry", "response":["What do you want to do together? Im... im open to anything!"], "background":"house.jpg"},            
            {"map":["accept invitation"], "action":"comfortgf1","img":"embarrassed", "response":["{0}! I.. I wanna f***. Im feeling horny af rn."], "background":"house.jpg"},
            {"map":["gaming"], "action":"comfortgf2","img":"image", "response":["I.. I wanna game."], "background":"house.jpg"},
            {"map":["netflix"], "action":"comfortgf3","img":"image", "response":["I.. I wanna watch netflix. "], "background":"house.jpg"},
            {"map":["movies"], "action":"comfortgf4","img":"image", "response":["I.. I wanna watch a movie."], "background":"house.jpg"},
            {"map":["comfortgf1", "comfortgf2", "comfortgf3","comfortgf4"], "action":"comfortgf","img":"angry", "response":["Hmm, I think I know what will cheer me up!"], "background":"house.jpg"},
            {"map":["great", "Im not feeling that way"], "action":"happygf","img":"image", "response":["Hey! How are you doing?"], "background":"house.jpg"},
            {"map":["yeah sure what?"], "action":"great","img":"embarrassed", "response":["I wanna go do something with you..."], "background":"house.jpg"},
            {"map":["gaming"], "action":"gaming","img":"image", "response":["{0}! Lets play a game! I havnt played with you in forever!!"], "background":"house.jpg"},
            {"map":["movies"], "action":"movies","img":"image", "response":["{0}! Lets watch a movie!"], "background":"house.jpg"},
            {"map":["netflix"], "action":"netflix","img":"image", "response":["{0}! Lets watch netflix"], "background":"house.jpg"},
            {"map":["kiss","hug"], "action":"attentionwant","img":"embarrassed", "response":["I want attention!"], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scaredgf","img":"angry", "response":["Im not sure how im gonna pay the rent."], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"angrygf","img":"angry", "response":["Im not having a good day. I just wanna go to sleep."], "background":"house.jpg"},
        ],

            [
            {"typename":"Yandere", "action":"none","img":"image", "response":"see you, {0}! I love you!", "background":"house.jpg"},
            {"map":["sad", "scared", "happy", "angry", "horny"], "action":"Im not feeling that way","img":"image", "response":["Im sorry, how are you feeling right now?"], "background":"house.jpg"},
            {"map":["accept invitation"], "action":"horny","img":"embarrassed", "response":["ohh, thats what you were feeling. Thats ok, I can help you out with that ;)"], "background":"house.jpg"},
            {"map":["leave","invite her to go do something"], "action":"end","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["No, Im fine", "Im not feeling that way"], "action":"sad","img":"image", "response":["It seems like you are sad. is that right? Thats too bad! is there anything I can do?"], "background":"house.jpg"},
            {"map":["lie on lap","dont lie on lap"], "action":"No, Im fine","img":"image", "response":["I cant give much, but I will support you will all Ive got! come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["leave","lie on lap"], "action":"dont lie on lap","img":"image", "response":["come on, just for a little while? come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["end"], "action":"lie on lap","img":"image","response": ["Hey. I know you can do it. I love you so much. That will never change. \n*You are my sunshine, My only sunshine\n You make me happy when skies are gray!\n You'll never know, dear, how much I love you!\n please dont take my sunshine away!*\n Did you like my voice? I hope so! *smooch*"], "background":"house.jpg"},
            {"map":["hug","kiss","Im really thankful for you!", "Im not feeling that way"], "action":"happy","img":"image", "response":["Thats amazing! im so happy for you!"], "background":"house.jpg"},
            {"map":["hug", "kiss", "end"], "action":"Im really thankful for you!","img":"image", "response":["aww, I love you so much! of course Id support you!"], "background":"house.jpg"},
            {"map":["end","kiss","invite her to go do something"], "action":"hug","img":"image", "response":["hmmm? You want a hug? of course!! *sqeezes*"], "background":"house.jpg"},
            {"map":["end","invite her to go do something"], "action":"kiss","img":"image", "response":["*mwah* I'll see you around! I love you!"], "background":"house.jpg"},
            {"map":["kiss"], "action":"Im really thankful for you!","img":"image", "response":["hey. I care about you!! Its only normal.."], "background":"house.jpg"},
            {"map":["walk away","lie on lap", "Im not feeling that way"], "action":"angry","img":"image", "response":["Hey. Im not sure if you are in the mood, you seem mad, or annoyed, but wanna rest on my lap?"], "background":"house.jpg"},
            {"map":["go with her", "dont follow"], "action":"walk away","img":"image", "response":["Hey I know just the thing! Follow me!"], "background":"house.jpg"},
            {"map":["end"], "action":"dont follow","img":"image", "response":["all right. I understand, Ill give you some time. If you wanna talk to me about anything, Im always available to you!"], "background":"house.jpg"},
            {"map":["leave","look at stars"], "action":"go with her","img":"image", "response":["this is the night sky! It looks nice, right? You can relax here. I find it nice gazing at the stars"], "background":"nightsky.jpg"},
            {"map":["end"], "action":"leave","img":"image", "response":["Im always ready to talk if you need me. I love you! bye!"], "background":"house.jpg"},
            {"map":["end"], "action":"look at stars","img":"image", "response":["Its nice right? Ill leave you be for now."], "background":"nightsky.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scared","img":"image", "response":["It sounds like you are scared. I know you are strong. You are also smart! if you cant handle it on your own, find someone to help you! You shouldnt always try to do things on your own!"], "background":"house.jpg"},
        
            {"map":["comfortgf"], "action":"sadgf","img":"sad", "response":["{0}, im feeling really sad! I dont like this! do something!"], "background":"house.jpg"},
            {"map":["gaming", "movies", "netflix", "horny"], "action":"invite her to go do something","img":"angry", "response":["What do you want to do together? Im... im open to anything!"], "background":"house.jpg"},            
            {"map":["accept invitation"], "action":"comfortgf1","img":"embarrassed", "response":["{0}! I.. I wanna f***. Im feeling horny af rn."], "background":"house.jpg"},
            {"map":["gaming"], "action":"comfortgf2","img":"image", "response":["I.. I wanna game."], "background":"house.jpg"},
            {"map":["netflix"], "action":"comfortgf3","img":"image", "response":["I.. I wanna watch netflix. "], "background":"house.jpg"},
            {"map":["movies"], "action":"comfortgf4","img":"image", "response":["I.. I wanna watch a movie."], "background":"house.jpg"},
            {"map":["comfortgf1", "comfortgf2", "comfortgf3","comfortgf4"], "action":"comfortgf","img":"angry", "response":["Hmm, I think I know what will cheer me up!"], "background":"house.jpg"},
            {"map":["great", "Im not feeling that way"], "action":"happygf","img":"image", "response":["Hey! How are you doing?"], "background":"house.jpg"},
            {"map":["yeah sure what?"], "action":"great","img":"embarrassed", "response":["I wanna go do something with you..."], "background":"house.jpg"},
            {"map":["gaming"], "action":"gaming","img":"image", "response":["{0}! Lets play a game! I havnt played with you in forever!!"], "background":"house.jpg"},
            {"map":["movies"], "action":"movies","img":"image", "response":["{0}! Lets watch a movie!"], "background":"house.jpg"},
            {"map":["netflix"], "action":"netflix","img":"image", "response":["{0}! Lets watch netflix"], "background":"house.jpg"},
            {"map":["kiss","hug"], "action":"attentionwant","img":"embarrassed", "response":["I want attention!"], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scaredgf","img":"angry", "response":["Im not sure how im gonna pay the rent."], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"angrygf","img":"angry", "response":["Im not having a good day. I just wanna go to sleep."], "background":"house.jpg"},
        ],

            [
            {"typename":"Dandere", "action":"none","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["sad", "scared", "happy", "angry", "horny"], "action":"Im not feeling that way","img":"image", "response":["Im sorry, how are you feeling right now?"], "background":"house.jpg"},
            {"map":["accept invitation"], "action":"horny","img":"embarrassed", "response":["ohh, thats what you were feeling. Thats ok, I can help you out with that ;)"], "background":"house.jpg"},
            {"map":["leave","invite her to go do something"], "action":"end","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["No, Im fine", "Im not feeling that way"], "action":"sad","img":"image", "response":["It seems like you are sad. is that right? Thats too bad! is there anything I can do?"], "background":"house.jpg"},
            {"map":["lie on lap","dont lie on lap"], "action":"No, Im fine","img":"image", "response":["I cant give much, but I will support you will all Ive got! come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["leave","lie on lap"], "action":"dont lie on lap","img":"image", "response":["come on, just for a little while? come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["end"], "action":"lie on lap","img":"image","response": ["Hey. I know you can do it. I love you so much. That will never change. \n*You are my sunshine, My only sunshine\n You make me happy when skies are gray!\n You'll never know, dear, how much I love you!\n please dont take my sunshine away!*\n Did you like my voice? I hope so! *smooch*"], "background":"house.jpg"},
            {"map":["hug","kiss","Im really thankful for you!", "Im not feeling that way"], "action":"happy","img":"image", "response":["Thats amazing! im so happy for you!"], "background":"house.jpg"},
            {"map":["hug", "kiss", "end"], "action":"Im really thankful for you!","img":"image", "response":["aww, I love you so much! of course Id support you!"], "background":"house.jpg"},
            {"map":["end","kiss","invite her to go do something"], "action":"hug","img":"image", "response":["hmmm? You want a hug? of course!! *sqeezes*"], "background":"house.jpg"},
            {"map":["end","invite her to go do something"], "action":"kiss","img":"image", "response":["*mwah* I'll see you around! I love you!"], "background":"house.jpg"},
            {"map":["kiss"], "action":"Im really thankful for you!","img":"image", "response":["hey. I care about you!! Its only normal.."], "background":"house.jpg"},
            {"map":["walk away","lie on lap", "Im not feeling that way"], "action":"angry","img":"image", "response":["Hey. Im not sure if you are in the mood, you seem mad, or annoyed, but wanna rest on my lap?"], "background":"house.jpg"},
            {"map":["go with her", "dont follow"], "action":"walk away","img":"image", "response":["Hey I know just the thing! Follow me!"], "background":"house.jpg"},
            {"map":["end"], "action":"dont follow","img":"image", "response":["all right. I understand, Ill give you some time. If you wanna talk to me about anything, Im always available to you!"], "background":"house.jpg"},
            {"map":["leave","look at stars"], "action":"go with her","img":"image", "response":["this is the night sky! It looks nice, right? You can relax here. I find it nice gazing at the stars"], "background":"nightsky.jpg"},
            {"map":["end"], "action":"leave","img":"image", "response":["Im always ready to talk if you need me. I love you! bye!"], "background":"house.jpg"},
            {"map":["end"], "action":"look at stars","img":"image", "response":["Its nice right? Ill leave you be for now."], "background":"nightsky.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scared","img":"image", "response":["It sounds like you are scared. I know you are strong. You are also smart! if you cant handle it on your own, find someone to help you! You shouldnt always try to do things on your own!"], "background":"house.jpg"},
        
            {"map":["comfortgf"], "action":"sadgf","img":"sad", "response":["{0}, im feeling really sad! I dont like this! do something!"], "background":"house.jpg"},
            {"map":["gaming", "movies", "netflix", "horny"], "action":"invite her to go do something","img":"angry", "response":["What do you want to do together? Im... im open to anything!"], "background":"house.jpg"},            
            {"map":["accept invitation"], "action":"comfortgf1","img":"embarrassed", "response":["{0}! I.. I wanna f***. Im feeling horny af rn."], "background":"house.jpg"},
            {"map":["gaming"], "action":"comfortgf2","img":"image", "response":["I.. I wanna game."], "background":"house.jpg"},
            {"map":["netflix"], "action":"comfortgf3","img":"image", "response":["I.. I wanna watch netflix. "], "background":"house.jpg"},
            {"map":["movies"], "action":"comfortgf4","img":"image", "response":["I.. I wanna watch a movie."], "background":"house.jpg"},
            {"map":["comfortgf1", "comfortgf2", "comfortgf3","comfortgf4"], "action":"comfortgf","img":"angry", "response":["Hmm, I think I know what will cheer me up!"], "background":"house.jpg"},
            {"map":["great", "Im not feeling that way"], "action":"happygf","img":"image", "response":["Hey! How are you doing?"], "background":"house.jpg"},
            {"map":["yeah sure what?"], "action":"great","img":"embarrassed", "response":["I wanna go do something with you..."], "background":"house.jpg"},
            {"map":["gaming"], "action":"gaming","img":"image", "response":["{0}! Lets play a game! I havnt played with you in forever!!"], "background":"house.jpg"},
            {"map":["movies"], "action":"movies","img":"image", "response":["{0}! Lets watch a movie!"], "background":"house.jpg"},
            {"map":["netflix"], "action":"netflix","img":"image", "response":["{0}! Lets watch netflix"], "background":"house.jpg"},
            {"map":["kiss","hug"], "action":"attentionwant","img":"embarrassed", "response":["I want attention!"], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scaredgf","img":"angry", "response":["Im not sure how im gonna pay the rent."], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"angrygf","img":"angry", "response":["Im not having a good day. I just wanna go to sleep."], "background":"house.jpg"},
        ],

            [
            {"typename":"Sadodere", "action":"none","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["sad", "scared", "happy", "angry", "horny"], "action":"Im not feeling that way","img":"image", "response":["Im sorry, how are you feeling right now?"], "background":"house.jpg"},
            {"map":["accept invitation"], "action":"horny","img":"embarrassed", "response":["ohh, thats what you were feeling. Thats ok, I can help you out with that ;)"], "background":"house.jpg"},
            {"map":["leave","invite her to go do something"], "action":"end","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["No, Im fine", "Im not feeling that way"], "action":"sad","img":"image", "response":["It seems like you are sad. is that right? Thats too bad! is there anything I can do?"], "background":"house.jpg"},
            {"map":["lie on lap","dont lie on lap"], "action":"No, Im fine","img":"image", "response":["I cant give much, but I will support you will all Ive got! come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["leave","lie on lap"], "action":"dont lie on lap","img":"image", "response":["come on, just for a little while? come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["end"], "action":"lie on lap","img":"image","response": ["Hey. I know you can do it. I love you so much. That will never change. \n*You are my sunshine, My only sunshine\n You make me happy when skies are gray!\n You'll never know, dear, how much I love you!\n please dont take my sunshine away!*\n Did you like my voice? I hope so! *smooch*"], "background":"house.jpg"},
            {"map":["hug","kiss","Im really thankful for you!", "Im not feeling that way"], "action":"happy","img":"image", "response":["Thats amazing! im so happy for you!"], "background":"house.jpg"},
            {"map":["hug", "kiss", "end"], "action":"Im really thankful for you!","img":"image", "response":["aww, I love you so much! of course Id support you!"], "background":"house.jpg"},
            {"map":["end","kiss","invite her to go do something"], "action":"hug","img":"image", "response":["hmmm? You want a hug? of course!! *sqeezes*"], "background":"house.jpg"},
            {"map":["end","invite her to go do something"], "action":"kiss","img":"image", "response":["*mwah* I'll see you around! I love you!"], "background":"house.jpg"},
            {"map":["kiss"], "action":"Im really thankful for you!","img":"image", "response":["hey. I care about you!! Its only normal.."], "background":"house.jpg"},
            {"map":["walk away","lie on lap", "Im not feeling that way"], "action":"angry","img":"image", "response":["Hey. Im not sure if you are in the mood, you seem mad, or annoyed, but wanna rest on my lap?"], "background":"house.jpg"},
            {"map":["go with her", "dont follow"], "action":"walk away","img":"image", "response":["Hey I know just the thing! Follow me!"], "background":"house.jpg"},
            {"map":["end"], "action":"dont follow","img":"image", "response":["all right. I understand, Ill give you some time. If you wanna talk to me about anything, Im always available to you!"], "background":"house.jpg"},
            {"map":["leave","look at stars"], "action":"go with her","img":"image", "response":["this is the night sky! It looks nice, right? You can relax here. I find it nice gazing at the stars"], "background":"nightsky.jpg"},
            {"map":["end"], "action":"leave","img":"image", "response":["Im always ready to talk if you need me. I love you! bye!"], "background":"house.jpg"},
            {"map":["end"], "action":"look at stars","img":"image", "response":["Its nice right? Ill leave you be for now."], "background":"nightsky.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scared","img":"image", "response":["It sounds like you are scared. I know you are strong. You are also smart! if you cant handle it on your own, find someone to help you! You shouldnt always try to do things on your own!"], "background":"house.jpg"},
        
            {"map":["comfortgf"], "action":"sadgf","img":"sad", "response":["{0}, im feeling really sad! I dont like this! do something!"], "background":"house.jpg"},
            {"map":["gaming", "movies", "netflix", "horny"], "action":"invite her to go do something","img":"angry", "response":["What do you want to do together? Im... im open to anything!"], "background":"house.jpg"},            
            {"map":["accept invitation"], "action":"comfortgf1","img":"embarrassed", "response":["{0}! I.. I wanna f***. Im feeling horny af rn."], "background":"house.jpg"},
            {"map":["gaming"], "action":"comfortgf2","img":"image", "response":["I.. I wanna game."], "background":"house.jpg"},
            {"map":["netflix"], "action":"comfortgf3","img":"image", "response":["I.. I wanna watch netflix. "], "background":"house.jpg"},
            {"map":["movies"], "action":"comfortgf4","img":"image", "response":["I.. I wanna watch a movie."], "background":"house.jpg"},
            {"map":["comfortgf1", "comfortgf2", "comfortgf3","comfortgf4"], "action":"comfortgf","img":"angry", "response":["Hmm, I think I know what will cheer me up!"], "background":"house.jpg"},
            {"map":["great", "Im not feeling that way"], "action":"happygf","img":"image", "response":["Hey! How are you doing?"], "background":"house.jpg"},
            {"map":["yeah sure what?"], "action":"great","img":"embarrassed", "response":["I wanna go do something with you..."], "background":"house.jpg"},
            {"map":["gaming"], "action":"gaming","img":"image", "response":["{0}! Lets play a game! I havnt played with you in forever!!"], "background":"house.jpg"},
            {"map":["movies"], "action":"movies","img":"image", "response":["{0}! Lets watch a movie!"], "background":"house.jpg"},
            {"map":["netflix"], "action":"netflix","img":"image", "response":["{0}! Lets watch netflix"], "background":"house.jpg"},
            {"map":["kiss","hug"], "action":"attentionwant","img":"embarrassed", "response":["I want attention!"], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scaredgf","img":"angry", "response":["Im not sure how im gonna pay the rent."], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"angrygf","img":"angry", "response":["Im not having a good day. I just wanna go to sleep."], "background":"house.jpg"},
        ],

            [
            {"typename":"Kuudere", "action":"none","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["sad", "scared", "happy", "angry", "horny"], "action":"Im not feeling that way","img":"image", "response":["Im sorry, how are you feeling right now?"], "background":"house.jpg"},
            {"map":["accept invitation"], "action":"horny","img":"embarrassed", "response":["ohh, thats what you were feeling. Thats ok, I can help you out with that ;)"], "background":"house.jpg"},
            {"map":["leave","invite her to go do something"], "action":"end","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["No, Im fine", "Im not feeling that way"], "action":"sad","img":"image", "response":["It seems like you are sad. is that right? Thats too bad! is there anything I can do?"], "background":"house.jpg"},
            {"map":["lie on lap","dont lie on lap"], "action":"No, Im fine","img":"image", "response":["I cant give much, but I will support you will all Ive got! come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["leave","lie on lap"], "action":"dont lie on lap","img":"image", "response":["come on, just for a little while? come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["end"], "action":"lie on lap","img":"image","response": ["Hey. I know you can do it. I love you so much. That will never change. \n*You are my sunshine, My only sunshine\n You make me happy when skies are gray!\n You'll never know, dear, how much I love you!\n please dont take my sunshine away!*\n Did you like my voice? I hope so! *smooch*"], "background":"house.jpg"},
            {"map":["hug","kiss","Im really thankful for you!", "Im not feeling that way"], "action":"happy","img":"image", "response":["Thats amazing! im so happy for you!"], "background":"house.jpg"},
            {"map":["hug", "kiss", "end"], "action":"Im really thankful for you!","img":"image", "response":["aww, I love you so much! of course Id support you!"], "background":"house.jpg"},
            {"map":["end","kiss","invite her to go do something"], "action":"hug","img":"image", "response":["hmmm? You want a hug? of course!! *sqeezes*"], "background":"house.jpg"},
            {"map":["end","invite her to go do something"], "action":"kiss","img":"image", "response":["*mwah* I'll see you around! I love you!"], "background":"house.jpg"},
            {"map":["kiss"], "action":"Im really thankful for you!","img":"image", "response":["hey. I care about you!! Its only normal.."], "background":"house.jpg"},
            {"map":["walk away","lie on lap", "Im not feeling that way"], "action":"angry","img":"image", "response":["Hey. Im not sure if you are in the mood, you seem mad, or annoyed, but wanna rest on my lap?"], "background":"house.jpg"},
            {"map":["go with her", "dont follow"], "action":"walk away","img":"image", "response":["Hey I know just the thing! Follow me!"], "background":"house.jpg"},
            {"map":["end"], "action":"dont follow","img":"image", "response":["all right. I understand, Ill give you some time. If you wanna talk to me about anything, Im always available to you!"], "background":"house.jpg"},
            {"map":["leave","look at stars"], "action":"go with her","img":"image", "response":["this is the night sky! It looks nice, right? You can relax here. I find it nice gazing at the stars"], "background":"nightsky.jpg"},
            {"map":["end"], "action":"leave","img":"image", "response":["Im always ready to talk if you need me. I love you! bye!"], "background":"house.jpg"},
            {"map":["end"], "action":"look at stars","img":"image", "response":["Its nice right? Ill leave you be for now."], "background":"nightsky.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scared","img":"image", "response":["It sounds like you are scared. I know you are strong. You are also smart! if you cant handle it on your own, find someone to help you! You shouldnt always try to do things on your own!"], "background":"house.jpg"},
        
            {"map":["comfortgf"], "action":"sadgf","img":"sad", "response":["{0}, im feeling really sad! I dont like this! do something!"], "background":"house.jpg"},
            {"map":["gaming", "movies", "netflix", "horny"], "action":"invite her to go do something","img":"angry", "response":["What do you want to do together? Im... im open to anything!"], "background":"house.jpg"},            
            {"map":["accept invitation"], "action":"comfortgf1","img":"embarrassed", "response":["{0}! I.. I wanna f***. Im feeling horny af rn."], "background":"house.jpg"},
            {"map":["gaming"], "action":"comfortgf2","img":"image", "response":["I.. I wanna game."], "background":"house.jpg"},
            {"map":["netflix"], "action":"comfortgf3","img":"image", "response":["I.. I wanna watch netflix. "], "background":"house.jpg"},
            {"map":["movies"], "action":"comfortgf4","img":"image", "response":["I.. I wanna watch a movie."], "background":"house.jpg"},
            {"map":["comfortgf1", "comfortgf2", "comfortgf3","comfortgf4"], "action":"comfortgf","img":"angry", "response":["Hmm, I think I know what will cheer me up!"], "background":"house.jpg"},
            {"map":["great", "Im not feeling that way"], "action":"happygf","img":"image", "response":["Hey! How are you doing?"], "background":"house.jpg"},
            {"map":["yeah sure what?"], "action":"great","img":"embarrassed", "response":["I wanna go do something with you..."], "background":"house.jpg"},
            {"map":["gaming"], "action":"gaming","img":"image", "response":["{0}! Lets play a game! I havnt played with you in forever!!"], "background":"house.jpg"},
            {"map":["movies"], "action":"movies","img":"image", "response":["{0}! Lets watch a movie!"], "background":"house.jpg"},
            {"map":["netflix"], "action":"netflix","img":"image", "response":["{0}! Lets watch netflix"], "background":"house.jpg"},
            {"map":["kiss","hug"], "action":"attentionwant","img":"embarrassed", "response":["I want attention!"], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scaredgf","img":"angry", "response":["Im not sure how im gonna pay the rent."], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"angrygf","img":"angry", "response":["Im not having a good day. I just wanna go to sleep."], "background":"house.jpg"},
        ],
            [
            {"typename":"Kamidere", "action":"none","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["sad", "scared", "happy", "angry", "horny"], "action":"Im not feeling that way","img":"image", "response":["Im sorry, how are you feeling right now?"], "background":"house.jpg"},
            {"map":["accept invitation"], "action":"horny","img":"embarrassed", "response":["ohh, thats what you were feeling. Thats ok, I can help you out with that ;)"], "background":"house.jpg"},
            {"map":["leave","invite her to go do something"], "action":"end","img":"image", "response":["see you, {0}! I love you!"], "background":"house.jpg"},
            {"map":["No, Im fine", "Im not feeling that way"], "action":"sad","img":"image", "response":["It seems like you are sad. is that right? Thats too bad! is there anything I can do?"], "background":"house.jpg"},
            {"map":["lie on lap","dont lie on lap"], "action":"No, Im fine","img":"image", "response":["I cant give much, but I will support you will all Ive got! come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["leave","lie on lap"], "action":"dont lie on lap","img":"image", "response":["come on, just for a little while? come here! *motions to rest on lap*"], "background":"house.jpg"},
            {"map":["end"], "action":"lie on lap","img":"image","response": ["Hey. I know you can do it. I love you so much. That will never change. \n*You are my sunshine, My only sunshine\n You make me happy when skies are gray!\n You'll never know, dear, how much I love you!\n please dont take my sunshine away!*\n Did you like my voice? I hope so! *smooch*"], "background":"house.jpg"},
            {"map":["hug","kiss","Im really thankful for you!", "Im not feeling that way"], "action":"happy","img":"image", "response":["Thats amazing! im so happy for you!"], "background":"house.jpg"},
            {"map":["hug", "kiss", "end"], "action":"Im really thankful for you!","img":"image", "response":["aww, I love you so much! of course Id support you!"], "background":"house.jpg"},
            {"map":["end","kiss","invite her to go do something"], "action":"hug","img":"image", "response":["hmmm? You want a hug? of course!! *sqeezes*"], "background":"house.jpg"},
            {"map":["end","invite her to go do something"], "action":"kiss","img":"image", "response":["*mwah* I'll see you around! I love you!"], "background":"house.jpg"},
            {"map":["kiss"], "action":"Im really thankful for you!","img":"image", "response":["hey. I care about you!! Its only normal.."], "background":"house.jpg"},
            {"map":["walk away","lie on lap", "Im not feeling that way"], "action":"angry","img":"image", "response":["Hey. Im not sure if you are in the mood, you seem mad, or annoyed, but wanna rest on my lap?"], "background":"house.jpg"},
            {"map":["go with her", "dont follow"], "action":"walk away","img":"image", "response":["Hey I know just the thing! Follow me!"], "background":"house.jpg"},
            {"map":["end"], "action":"dont follow","img":"image", "response":["all right. I understand, Ill give you some time. If you wanna talk to me about anything, Im always available to you!"], "background":"house.jpg"},
            {"map":["leave","look at stars"], "action":"go with her","img":"image", "response":["this is the night sky! It looks nice, right? You can relax here. I find it nice gazing at the stars"], "background":"nightsky.jpg"},
            {"map":["end"], "action":"leave","img":"image", "response":["Im always ready to talk if you need me. I love you! bye!"], "background":"house.jpg"},
            {"map":["end"], "action":"look at stars","img":"image", "response":["Its nice right? Ill leave you be for now."], "background":"nightsky.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scared","img":"image", "response":["It sounds like you are scared. I know you are strong. You are also smart! if you cant handle it on your own, find someone to help you! You shouldnt always try to do things on your own!"], "background":"house.jpg"},
        
            {"map":["comfortgf"], "action":"sadgf","img":"sad", "response":["{0}, im feeling really sad! I dont like this! do something!"], "background":"house.jpg"},
            {"map":["gaming", "movies", "netflix", "horny"], "action":"invite her to go do something","img":"angry", "response":["What do you want to do together? Im... im open to anything!"]},            
            {"map":["accept invitation"], "action":"comfortgf1","img":"embarrassed", "response":["{0}! I.. I wanna f***. Im feeling horny af rn."], "background":"house.jpg"},
            {"map":["gaming"], "action":"comfortgf2","img":"image", "response":["I.. I wanna game."], "background":"house.jpg"},
            {"map":["netflix"], "action":"comfortgf3","img":"image", "response":["I.. I wanna watch netflix. "], "background":"house.jpg"},
            {"map":["movies"], "action":"comfortgf4","img":"image", "response":["I.. I wanna watch a movie."], "background":"house.jpg"},
            {"map":["comfortgf1", "comfortgf2", "comfortgf3","comfortgf4"], "action":"comfortgf","img":"angry", "response":["Hmm, I think I know what will cheer me up!"], "background":"house.jpg"},
            {"map":["great", "Im not feeling that way"], "action":"happygf","img":"image", "response":["Hey! How are you doing?"], "background":"house.jpg"},
            {"map":["yeah sure what?"], "action":"great","img":"embarrassed", "response":["I wanna go do something with you..."], "background":"house.jpg"},
            {"map":["gaming"], "action":"gaming","img":"image", "response":["{0}! Lets play a game! I havnt played with you in forever!!"], "background":"house.jpg"},
            {"map":["movies"], "action":"movies","img":"image", "response":["{0}! Lets watch a movie!"], "background":"house.jpg"},
            {"map":["netflix"], "action":"netflix","img":"image", "response":["{0}! Lets watch netflix"], "background":"house.jpg"},
            {"map":["kiss","hug"], "action":"attentionwant","img":"embarrassed", "response":["I want attention!"], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"scaredgf","img":"angry", "response":["Im not sure how im gonna pay the rent."], "background":"house.jpg"},
            {"map":["end", "Im not feeling that way"], "action":"angrygf","img":"angry", "response":["Im not having a good day. I just wanna go to sleep."], "background":"house.jpg"},
        ],
        ]

def getBoinkResponse():
    return [
            {"typename":"Tsundere",
             "start":["Hello!", "again?"], 
             "kiss":["..thanks! You are really good!", "aww, I love you too!"], 
             "pin down":["eh? What are you doing?", "again pin dowN"], 
             "fondle oppai":["*oh* t. That feels really good!", "again? I dont mind though..."], 
             "suck oppai":["*ahh* How do you like my boobs?", "You really like my boobs, dont you.."], 
             "finger vegana":["stop.. im really sensitive there!", "I think I might reach my limit! Its amazing!"],
             "lick vegana":["How does it taste?", "Youre a greedy boy, {0}, You keep coming back for more, huh?"], 
             "bite":["awww", "What do you think of my skin?"], 
             "insert pp":["oh! Its so big!, It feels like heaven!", "againinsert"], 
             "climax": "*That felt amazing. I love you so so much. I...\n I want 3 kids."},

            {"typename":"Yandere", 
            "start":["What do you plan to do to me today? ;)", "again?"], 
            "kiss":["huh, feeling horny are you?", "I never get tired of kissing you <3"], 
            "pin down":["Oh, its new seeing you with the initiative.. I like it!", "again pin dowN"], 
            "fondle oppai":["These tits are yours. Do you think they are bouncy?", "You really like my tits huh?"], 
            "suck oppai":["Please feel free to suck on my milkers anytime,", "Came back for more huh?"], 
            "finger vegana":["That feels sooo good", "This feels great.."],
            "lick vegana":["Im sensitive there, but go on. How does this fresh pussy taste, {0}?", "My pussy tasted so good, you came back for more, huh?"], 
            "bite":["aww, I love you too!", "bite me moree"], 
            "insert pp":["oh my! You feel even better than I imagined!! I cant tell you how long ive been waiting for this!", "againinsert"], 
            "climax": "That felt great Lets do this more, and more and more!!!!!"},            
            
            {"typename":"Dandere", 
            "start":["..oh hi!", "again?"], 
            "kiss":[".. *blushes* thank you..", "I love you too.."], 
            "pin down":[".. what are you doing?", "again pin dowN"], 
            "fondle oppai":["oh my.. that feels so good.", "*mph* im sensitive."], 
            "suck oppai":["that feels so good! I really like this!", "keep going!"], 
            "finger vegana":["..im sensitive there! I.. might not last long", "*im really sensitive there, {0}-kun"],
            "lick vegana":["{0}-kun is licking my..!", "it feels great!"], 
            "bite":["i want to bite you too!", "let me bite you! *bites back*"], 
            "insert pp":["i..its so big!", "againinsert"], 
            "climax": "{0}-kun, that felt amazing!"},  

            {"typename":"Kuudere", 
            "start":["Hello.", "again?"], 
            "kiss":["continue,", "I like your lips."], 
            "pin down":["Pinning me down now?", "again pin dowN"], 
            "fondle oppai":["You like these milkers?", "They are bouncy arent they?"], 
            "suck oppai":["I like this feeling. Keep sucking", "coming back to my milkers, You must like them?"], 
            "finger vegana":["Im sensitive there. I might come!", "I really like that!"],
            "lick vegana":["oh, This feels great", "amazing!!!"], 
            "bite":["marking me huh? thats pretty kinky.", "*bites back*"], 
            "insert pp":["its so big!!", "againinsert"], 
            "climax": "You are great {0}, I love you so much!"},  

            {"typename":"Sadodere", 
            "start":["Oh hey, {0}", "again?"], 
            "kiss":["hmm??? arent you taking initiative!", "these lips really turn you on huh?"], 
            "pin down":["I never knew this part of you, {0}!", "again pin dowN"], 
            "fondle oppai":["You go for my tits huh? Pervert!!!!!! hahahahaha, im joking, Go on,", "are my tits that bouncy?"], 
            "suck oppai":["ara ara, how do my tits taste?", "You like that dont you?"], 
            "finger vegana":["Is my pussy wetter than you imagined?", "hahaha! It feels great!!"],
            "lick vegana":["How does this pussy taste?", "damn, it tastes good huh?"], 
            "bite":["ooh, {0} is marking me as his! hahaha pervert!! but.. go on. I like it.", "*smacks*"], 
            "insert pp":["you finally took it out!", "againinsert"], 
            "climax": "Hey, {0}, youre not that bad. I.. I want 4 kids."},  

            {"typename":"Sweet", 
            "start":["oh hello {0}-kun!", "again?"], 
            "kiss":["huh, you really are taking initiative today!! <3", "I love you so so much!"], 
            "pin down":["*ehhh?* wha.. what are you doing {0}-kun?\n aha! I see how it is, go on!", "again pin dowN"], 
            "fondle oppai":["How do these feel? are they bouncy?", "that feels great!"], 
            "suck oppai":["{0}-kun, you really like my boobs, dont you?", "*mph* keep sucking!"], 
            "finger vegana":["{0}-kun.... Im really sensitive there!", "ah.. stop, I might come<3"],
            "lick vegana":["ohh gosh, thats amazing!", "You came back for more huh? Does my pussy taste that good to you?"], 
            "bite":["*ahh*, I love you too! *bites back*", "*ahhh*"], 
            "insert pp":["Its so big!!!! Im so happy! Its even better than I thought!!!!", "againinsert"], 
            "climax": "You are amazing {0}-kun. I love you so so so much! I wanna be with you forever! I wanna grow old together with you, {0}-kun!"},  

            {"typename":"Kamidere", 
            "start":["Oh, hello, {0}", "again?"], 
            "kiss":["*mph* haha, nice!", "I love you too <3"], 
            "pin down":["*hmmm?* what are you planning on doing? <3", "again pin dowN"], 
            "fondle oppai":["*ohh* How do me breasts feel? are they satisfactory?", "You really like my breasts dont you?"], 
            "suck oppai":["hmm, keep going <3", "oh god, this feels soo good."], 
            "finger vegana":["I am sensitive there, {0}", "You really want me to orgasm huh?"],
            "lick vegana":["How does this fresh taint taste? Is is salty? <3", "mmm, coming back for more huh?"], 
            "bite":["awww", "Again?"], 
            "insert pp":["Its.. bigger than I expected..", "againinsert"], 
            "climax": "That felt great, {0}, I cant wait to spend time with you again <3"},  
        ]

def getGfTypes():
    return [
            {"typename":"Tsundere", "letter":"a","textresponse": ["You are really bad at that, you know?", "That was fine, I guess.", "That was... Nice. t...tthank you."], "movieresponse": "thanks for taking me out!", "netflixresponse":["That show sucked lmao","that show was ok","Netflix is fun with you!"], "hugresponse":"I love you too! *squeezes*", "kissresponse": "... that was sudden. Youre a great kisser. I wouldnt mind another one <3", "proposeresponse": "YESSS!! YESS I LOVE YOU SOO MUCH {0}!!!"},
            {"typename":"Yandere", "letter":"b", "textresponse": ["maybe you should try harder? I will support you in any way I can.", "Thank you for the text.", "Thank you for the text. ily very much." ], "movieresponse": "I want to see more movies with you!", "netflixresponse":["That show sucked lmao","that show was ok","Netflix is fun with you!"], "hugresponse":"Dont move. I wanna stay like this for a few more hours.", "kissresponse": "stop. Dont leave. Kiss me again. Again. And again...", "proposeresponse": "of course Ill marry you!! i want to spend all my time with you! {0}!!!"},
            {"typename":"Dandere", "letter":"c", "textresponse": [".. thanks, but.. please try harder next time!","...I appreciate the text.", "Thank you for the text... I love you too."], "movieresponse": "Thank... you.. for taking me out!" , "netflixresponse":["That show sucked lmao","that show was ok","Netflix is fun with you!"], "hugresponse":"T.. thank you.", "kissresponse": "...thanks.", "proposeresponse": ".. of course!!!! I love you so much, {0}!!!"},
            {"typename":"Kuudere", "letter":"d", "textresponse": ["That was terrible.", "Decent at best.", "This is great. I love you very much."], "movieresponse": "That was a good movie.", "netflixresponse":["That show sucked lmao","that show was ok","Netflix is fun with you!"], "hugresponse":"Squeeze me more. i like this feeling.", "kissresponse": "Kiss me again. I like that feeling", "proposeresponse": "marry you? yeh sure ig. I guess you are now my fiance, {0}!!!"},
            {"typename":"Sweet", "letter":"e", "textresponse": ["Thank you! but try a little bit better next time?", "Thank you! I appreciate what you do!!", "This is amazing!!! Thank you! ily so so much!"], "movieresponse": "woow! that was great! we should do this more often!!", "netflixresponse":["That show sucked lmao","that show was ok","Netflix is fun with you!"], "hugresponse":"aww thanks! I love you too!! *squeezes* I dont ever want to lose you!", "kissresponse": "... that was sudden. Youre a great kisser. I wouldnt mind another one <3 I love you so much!", "proposeresponse": "YES! Of course I want to marry you! I want to spend time with you, Have kids, Grow old together. I love you so much, {0}!!!"},
            {"typename":"Sadodere", "letter":"f", "textresponse": ["You are really bad at texting!! I find it amusing.", "That was a decent text! Only Decent though.", "Good job! I am satisfied with that."], "movieresponse": "Isnt that something? Taking your girlfriend out to watch a movie.", "netflixresponse":["That show sucked lmao","that show was ok","Netflix is fun with you!"], "hugresponse":"huh? youre hugging me? Fine. Ill allow it. Pervert.", "kissresponse": ".. AH.. AHAHAHA did you just kiss me? pervert <3", "proposeresponse": "Marry you? Haha, Of course. I love you, {0} <3"},
            {"typename":"Kamidere", "letter":"g", "textresponse": ["Your texting skill is poor; It can be improved though.", "That was good effort. However, your text was only decent.", "Excellent. I appreciate it.❤️"], "movieresponse": "Thank you for the invitation. I greatly appreciate it❤️", "netflixresponse":["That show sucked lmao","that show was ok","Netflix is fun with you!"], "hugresponse":"Thank you. I love your embrace.", "kissresponse": "Youre great at that <3.", "proposeresponse": "{0}. Regarding your marriage proposal, I gratefully accept. words cant describe how much you mean to me. I want to spend the rest of my life with you<3."},
        ]      

def getTypeComplaint():
    return [
            {"typename": "Tsundere", 
            "strategy": "I dont really like strategy. but I guess its fine."},

            {"typename": "Yandere", 
            "strategy": "strategy isnt my forte. It isnt necessary either. I know everything about you already <3"},

            {"typename": "Dandere", 
            "strategy": "...I would prefer another genre.."},

            {"typename": "Kuudere", 
            "strategy": "strategy isnt fun."},

            {"typename": "Sweet", 
            "strategy": "I really appreciate the thought, but I think we could do another genre?"},

            {"typename": "Sadodere", 
            "strategy": "i dont like strategy. Its gross."},

            {"typename": "Kamidere", 
            "strategy": "I dont enjoy strategy games. They create an uptight atmosphere, that isnt ideal for our relationship."},

        ]

def getGfGamingResponse():
    return [
            {"typename":"Tsundere", "poor":"That wasnt really fun.", "medium": "I had a good time i guess, but thats to be expected! its a game after all.", "good":"Again! Lets play again! That was really nice!"},
            {"typename":"Yandere", "poor":"I will try to do better next time.", "medium": "That was mediocre at best. Developers are terrible!", "good":"That was amazing. please get more love points so you can do me <3."},
            {"typename":"Dandere", "poor":"I.. think we should try again?", "medium": "that was fine!", "good":"I... really enjoyed that! Lets do it again soon?"},
            {"typename":"Kuudere", "poor":"You are really bad! Its alright though.", "medium": "That was ok i guess, You arent really the best at this game are you?", "good":"You are pretty good actually."},
            {"typename":"Sweet", "poor":"You are really bad! Its alright though.", "medium": "That was ok i guess, You arent really the best at this game are you?", "good":"You are pretty good actually."},
            {"typename":"Sadodere", "poor":"You are really bad! Its alright though.", "medium": "That was ok i guess, You arent really the best at this game are you?", "good":"You are pretty good actually."},
            {"typename":"Kamidere", "poor":"You are really bad! Its alright though.", "medium": "That was ok i guess, You arent really the best at this game are you?", "good":"You are pretty good actually."},

        ]

def getTypeGenrePraise():
    return [
            {"typename": "Tsundere", 
            "strategy": "I really like strategy!",
            "horror":"That wasnt scary at all!", 
            "fps":"I love FPS games!", 
            "creativity":"I think Creativity games are the best!",
            "adventure":"I think Adventure games are the best!", 
            "animation":"The animation was great! I think the creators did an amazing job dont you think?", 
            "action":"Action is great!!"},

            {"typename": "Yandere", 
            "strategy": "I love this uptight atmosphere.",
            "horror":"That wasnt scary at all!", 
            "fps":"I love FPS games!", 
            "creativity":"I think Creativity games are the best!",
            "adventure":"I think Adventure games are the best!", 
            "animation":"The animation was great! I think the creators did an amazing job dont you think?", 
            "action":"Action is great!!"},

            {"typename": "Dandere", 
            "strategy": "...I like this genre!",
            "horror":"That wasnt scary at all!", 
            "fps":"I love FPS games!", 
            "creativity":"I think Creativity games are the best!",
            "adventure":"I think Adventure games are the best!", 
            "animation":"The animation was great! I think the creators did an amazing job dont you think?", 
            "action":"Action is great!!"},

            {"typename": "Kuudere", 
            "strategy": "strategy is fun.",
            "horror":"That wasnt scary at all!", 
            "fps":"I love FPS games!", 
            "creativity":"I think Creativity games are the best!",
            "adventure":"I think Adventure games are the best!", 
            "animation":"The animation was great! I think the creators did an amazing job dont you think?", 
            "action":"Action is great!!"},

            {"typename": "Sweet", 
            "strategy": "woow! this is really fun! strategy is really fun!!",
            "horror":"That wasnt scary at all!", 
            "fps":"I love FPS games!", 
            "creativity":"I think Creativity games are the best!",
            "adventure":"I think Adventure games are the best!", 
            "animation":"The animation was great! I think the creators did an amazing job dont you think?", 
            "action":"Action is great!!"},

            {"typename": "Sadodere",
            "strategy": "strategy. That sounds so much like you!",
            "horror":"That wasnt scary at all!",
            "fps":"I love FPS games!",
            "creativity":"I think Creativity games are the best!",
            "adventure":"I think Adventure games are the best!",
            "animation":"The animation was great! I think the creators did an amazing job dont you think?", 
            "action":"Action is great!!"},

            {"typename": "Kamidere", 
            "strategy": "I enjoy strategy. I think its incredibly vital to act logically in a relationship.",
            "horror":"That wasnt scary at all!", 
            "fps":"I love FPS games!", 
            "creativity":"I think Creativity games are the best!",
            "adventure":"I think Adventure games are the best!", 
            "animation":"The animation was great! I think the creators did an amazing job dont you think?", 
            "action":"Action is great!!"},

        ]


def getTypePraise():
    return [
            {"typename": "Tsundere", "text": "we should text more often.. I care about you a lot.", "gaming":"I really like playing games!", "movies":"I love movies. ", "relaxing":"I love this quality time with you!"},
            {"typename": "Yandere", "text": "Lets text more! I want to know everything about you<3", "gaming":"Gaming is incredibly fun with you. We should do this more often.", "movies":"I love movies. ", "relaxing":"I love this quality time with you!"},
            {"typename": "Dandere", "text": "...lets do this more often?", "gaming":"I.. really enjoyed that!! maybe we could play more often?", "movies":"I love movies. ", "relaxing":"I love this quality time with you!"},
            {"typename": "Kuudere", "text": "Text me more often.", "gaming":"That was fun. We will play more often from now on.", "movies":"I love movies. ", "relaxing":"I love this quality time with you!"},
            {"typename": "Sweet", "text": "I love the text! Thank you for keeping me in touch!", "gaming":"wooW! Im so happy we could play games together! Im glad you remembered that I like gaming!", "movies":"I love movies. I love you so much!! ", "relaxing":"I love this quality time with you!"},
            {"typename": "Sadodere", "text": "I found that satisfactory! dont get any weird ideas, though!", "gaming":"I wonder how you knew I like gaming? Pervert!!", "movies":"I love movies. ", "relaxing":"I love this quality time with you!"},
            {"typename": "Kamidere", "text": "I found that enjoyable. Texting is in fact, the most practical form of communication. I appreciate you.", "gaming":"I found that enjoyable. Thank you for this. We should play more often.<3", "movies":"I love movies. ", "relaxing":"I love this quality time with you!"},

        ]              



def getGFimage(p:discord.Member,emotion:str="image"):
    emotions=["embarrassed", "horny","surprised","climax", "image", "bed", "angry", "fear", "sad", "dissapointed"]
    gfval = mulah.find_one({"id":p.id}, {"gf"})["gf"]
    emotion = emotion.lower()
    if emotion in emotions:
        try:
            return gfval[emotion]
        except:
            try:
                return gfval["image"]
            except:
                raise noImageError(p)
    else:
        raise noImageError(p)

def addIrrelevantWarning(em:discord.Embed):
    em.add_field(name="Irrelevance warning.", value="It appears you are going off topic. Dont.")

openai.organization = "org-6cx7PCsPB7dbTOcOu2oI6nYX"
openai.api_key = "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"

def gpt3Classification(query, examples, labels):
    a=openai.Classification.create(
        search_model="ada",
        model="curie",
        examples=examples,
        query=query,
        labels=labels,
    )
    return a["label"]
def classifyGFText(prompt):
    labels = ["good","bad","decent"]
    examples = [
        ["I love you", "good"],
        ["Why dont you do this correctly?", "bad"],
        ["where do you want to eat?", "decent"],
        ["Im doing fine.", "decent"],
        ["You are so pretty", "good"],
        ["you look fat", "bad"]
    ]
    return gpt3Classification(prompt, examples=examples, labels=labels)

def classifyGFBoinking(prompt):
    labels = [
        "kiss",
        "hug",
        "breast groping",
        "pinning down",
        "about to climax",
        "climax",
        "filler",
        "irrelevant",
    ]

    examples = [
        ["*kisses passionately", "kiss"],
        ["*touches lips*", "kiss"],
        ["your lips are great", "kiss"],
        ["*pulls you closer*", "hug"],
        ["*hugs you tightly*","hug"],
        ["Hug me really really tight", "hug"],
        ["your tits are great", "breast groping"],
        ["your boobs are the best", "breast groping"],
        ["*grabs breasts*", "breast groping"],
        ["*pins down*", "pinning down"],
        ["*pushes you down*", "pinning down"],

        ["im cumming", "climax"],
        ["OHHHHH IM COMING", "climax"],
        ["*cums*", "climax"],
        ["*nuts inside*", "climax"],
        ["Im nutting", "climax"],

        ["im about to cum", "about to climax"],
        ["im going to nut", "about to climax"],
        ["IM GOING TO CUM", "about to climax"],
        ["Im going to climax", "about to climax"],
        ["oooh, im about to cum", "about to climax"],

        ["takes you to the store", "irrelevant"],
        ["How are your grades", "irrelevant"],
        ["What window issues are you having", "irrelevant"],
        ["Valorant is hard", "irrelevant"],
        ["gunfight in afghanistan", "irrelevant"],
        ["Spiderman is terrible, dont you think", "irrelevant"],

        ["ok", "filler"],
        ["lets do that then", "filler"],
        ["take your time", "filler"],
        ["im sorry i guess", "filler"],
        ["I love you too", "filler"],
        ["sure", "filler"]
    ]
    return gpt3Classification(prompt, examples=examples, labels=labels)



def classifyGFTalking(prompt):
    labels = [
        "kiss",
        "hug",
        "filler",
        "over",
        "nsfw"
    ]

    examples = [
        ["*kisses passionately", "kiss"],
        ["*touches lips*", "kiss"],
        ["your lips are great", "kiss"],
        ["*pulls you closer*", "hug"],
        ["*hugs you tightly*","hug"],
        ["Hug me really really tight", "hug"],

        ["ok", "filler"],
        ["lets do that then", "filler"],
        ["take your time", "filler"],
        ["im sorry i guess", "filler"],
        ["I love you too", "filler"],
        ["sure", "filler"],
        ["yeah thats prettty cool", "filler"],
        ["I hate when that happens", "filler"],
        ["yeah that sucks", "filler"],

        ["your boobs are the best", "nsfw"],
        ["*grabs breasts*", "nsfw"],
        ["*pins down*", "nsfw"],
        ["*pushes you down*", "nsfw"],

        ["im cumming", "nsfw"],
        ["OHHHHH IM COMING", "nsfw"],
        ["*cums*", "nsfw"],
        ["*nuts inside*", "nsfw"],
        ["Im nutting", "nsfw"],

        ["im about to cum", "nsfw"],
        ["im going to nut", "nsfw"],
        ["IM GOING TO CUM", "nsfw"],
        ["Im going to climax", "nsfw"],
        ["oooh, im about to cum", "nsfw"],



        ["bye", "over"],
        ["cya", "over"],
        ["Ill see you soon!", "over"],
        ["I got to go!", "over"],
        ["bye","over"]
    ]
    return gpt3Classification(prompt, examples=examples, labels=labels)

def getModel(id):
    gf = mulah.find_one({"id":id}, {"gf"})["gf"]
    gftype = gf["type"]
    gfdict = {
        "Tsundere":"curie:ft-sentientproductions-2021-12-29-17-58-33",
        "Yandere":"curie:ft-sentientproductions-2021-12-29-18-01-46",
        "Dandere":"curie:ft-sentientproductions-2021-12-29-18-05-49",
        "Kuudere":"curie:ft-sentientproductions-2021-12-29-18-08-10",
        "Sweet":"curie:ft-sentientproductions-2021-12-29-18-10-57",
        "Sadodere":"curie:ft-sentientproductions-2021-12-29-18-13-08",
        "Kamidere":"curie:ft-sentientproductions-2021-12-29-18-15-14"
    }
    return gfdict[gftype]

def classifyGFEmotion(prompt, filterNSFW=True):
    nsfw = ["horny","bed","climax"]
    examples = [
        ["sighs", "dissapointed"],
        ["Why are you like this? Its so annoying", "angry"],
        ["I love you", "image"],
        ["that was unexpected", "surprised"],
        ["stop... you are embarrassing me", "embarrassed"],
        ["Every time I want to help you, you push me away. It makes me sad.", "sad"],
        ["thats pretty scary", "fear"],
        ["You have a nice cock", "horny"],
        ["put it in me", "horny"],
        ["ahhh, you are really good at doing this!", "horny"],
        ["that was great. I want to marry you.", "bed"],
        ["that was great. we should fuck more often", "bed"],
        ["AHHHHHHHHhhhhhhhhhhhhhh im coming", "climax"],
        ["Im cumming!!", "climax"],
        ["oooohhhhhhh im climaxing!", "climax"]
    ]

    labels = [
        "dissapointed",
        "angry",
        "image",
        "surprised",
        "embarrassed",
        "sad",
        "fear",
        "horny",
        "bed",
        "climax"
    ]

    if filterNSFW:
        examples = [x for x in examples if x[1] not in nsfw]
        labels = [x for x in labels if x not in nsfw]
    return gpt3Classification(
        query=prompt,
        examples=examples,
        labels=labels
    )

def getGFresponse(prompt,person:discord.Member):
    model = getModel(person.id)
    background_prompt = chat.getprompt(person)
    girlfriend = mulah.find_one({"id":person.id}, {"gf"})["gf"]
    final = gpt3completion(
        background_prompt+"\n%s:%s\n%s:"%(person.display_name, prompt, girlfriend["name"]),
        model,
        person.display_name,
        girlfriend["name"]
    )
    return final

def gpt3completion(prompt, model, you,gf):
    openai.Engine.retrieve("davinci")
    z = openai.Completion.create(
    prompt=prompt,
    model=model,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=["\n", "%s:"%(you), "%s:"%(gf)]
    )
    return z["choices"][0]["text"]


class chat(object):
    def __init__(self, chatlog, you, other, model):
        self.chatlog = chatlog
        self.you = you
        self.other = other
        self.model = model
    
    def ask(self,question):
        response = openai.Completion.create(
            model=self.model,
            prompt=self.chatlog +f"\n{self.you}:" +question + f"\n{self.other}:",
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=["\n", f"{self.you}:", f"{self.other}:"]
        )
        answer = response["choices"][0]["text"]
        self.chatlog += f"\n{self.you}:{question}"+ f"\n{self.other}:{answer}"
        return answer
    @staticmethod
    def getprompt(user:discord.Member):
        #{
        #    "kisses":0,#
        #    "boinks":0,#
        #    "dates":0,#
         #   "hugs":0,#
          #  "games":0,#
           # "text":0,#
           # "netflix":0,#
         #   "movies":0,
         #   "start": date.today().strftime("%B %d, %Y"),
        #}},
        gf = mulah.find_one({"id":user.id}, {"gf"})["gf"]
        gfdata = mulah.find_one({"id":user.id}, {"gfdata"})["gfdata"]
        status = "lover"
        if gf["tier"] == 4:
            status = "fiance"
        prompt = "The following is a conversation between a %s girl whose name is %s and her %s, whose name is %s."%(
            gf["type"], gf["name"],status, user.display_name
        )

        prompt+=" %s has been dating %s since %s. They have kissed %s times, hugged %s times, had sex %s times, played games %s times, texted %s times, watched netlix together %s times, and watched movies %s times"%(
            gf["name"], user.display_name, gfdata["start"], gfdata["kisses"], gfdata["hugs"], gfdata["boinks"], gfdata["games"], gfdata["text"], gfdata["netflix"], gfdata["movies"]
        )

        prompt+=" %s's hobby is %s. Her favorite genre is %s, her least favorite is %s. Her favorite subject is %s."%(
            gf["name"], gf["likes"], gf["favorite genre"], gf["dislikes"], gf["favorite subject"]
        )
        return prompt









def getFunCommands():
    return ">  `pp`,`roll`,`rate`,`wisdom`, `rickroll`, `yomomma`, `8ball`, `animepic`, `cookie`, `coffee`, `story`"

def getModCommands():
    return "> `automod`,`ban`,`kick`,`mute`,`unmute`,`block`,`unblock`,`softban`, `swear`, `announce`,`suggest`, `swearlb`"

def getSolveCommands():
    return "> `hangman`, `scramble`"

def getUtilityCommands():
    return "> `snipe`, `esnipe`, `poll`, `timer`,`clean`, `choose`,`userinfo`,`serverinfo`,`channellinfo`,`permissions`"

def getGamesCommands():
    return "> `mcstatus`, `mcskin`"

def getVcCommands():
    return "> `p`,`pause`,`leave`,`resume`,`stop`"
def getMathCommands():
    return "> `gcf`,`points`, `simplify`, `herons`, `hardsolve`, `softsolve`"
def getWebCommands():
    return "> `question`, `imdb`reddit (group):`sub`,`reset`,`set` "
def getLevelCommands():
    return "> `rank`, `ranklb`"
def getEconomyCommands():
    return "> `rob`,`work`,`profile`,`worklist`,`apply`,`fish`,`hunt`,`mine`,`farm`,`chop`,`sell`,`craft`,`upgradepoint`,`send`,`achievement`,`achievement`,`balance`,`richlb`,`shop`,`use`, `give`,`gamestats`,`dep`,`withdraw`,`buy`,`inv`,`beg`,`watchlist`,`clearwatchlist` pc (group):`build`,`stats`,`addram`,`install`,`dismantle`,`play`"""

def getGfCommands():
    return "> `getgf`,`gfstats`,`breakup`, gf (group):`image`,`netflix`,`hug`,`kiss`,`boink`,`propose`,`date`,`movies`,`text`,`gaming`,`talk`"

def getImageCommands():
    return "> `avatar`,`animeface`,`caption`,`ddlc`,`blurpify`,`phcomment`,`toxicity`,`weebify`,`tweet`,`nichijou`,`threats`,`bodypillow`,`baguette`,`deepfry`,`clyde`,`ship`,`lolice`,`fact`,captcha`,`trash`,`whowouldwin`,`awooify`,`changemymind`,`magik`,`jpeg`,`gif`,`cat`,`dog`,`iphonex`,`kannagen`,`minesweeper`,`wanted`,`abouttocry`,`animepic`"

def getDuelsCommands():
    return "> `duel`,`equip`,`upgrade`,`begin`"

def getSettingsCommands():
    return "> `settings`,config (group): `badword`,`announcement`,`suggestion`,`setprefix`"




##-------------------------------------------------------------------ASYNC FUNCTS
async def Imdb(ctx, moviee):
    await ctx.trigger_typing()
    moviesDB=IMDb()
    movies = moviesDB.search_movie(moviee)
    print(movies)
    movieID = movies[0].getID()
    movie = moviesDB.get_movie(movieID)

            
    yt = YoutubeSearch(str(movie)+" trailer", max_results=1).to_json()
    yt_id = str(json.loads(yt)['videos'][0]['id'])
    yt_url = 'https://www.youtube.com/watch?v='+yt_id
    newyt = YoutubeSearch(str(movie)+" opening", max_results=1).to_json()
    newytid = str(json.loads(newyt)['videos'][0]['id'])
    thumnail_url = "https://img.youtube.com/vi/%s/maxresdefault.jpg"%(newytid)
    try:
        embed = discord.Embed(title = "%s, (%s)"%(movie, movie["year"]),url = yt_url,description = " Genre:%s"%(movie["genres"]), color = ctx.author.color)
    except:
        embed = discord.Embed(title = "%s"%(movie),url = yt_url,description = " Genre:%s"%(movie["genres"]), color = ctx.author.color)
    try:
        embed.add_field(name = "Synopsis:", value = "%s"%(str(moviesDB.get_movie_synopsis(movieID)["data"]["plot"][0])))
    except:
        pass
    embed.set_image(url = thumnail_url)
    embed.add_field(name = "Trailer", value = yt_url, inline=False)

    listofdirectories = ["rating"]
    for x in listofdirectories:
        try:
            embed.add_field(name = x, value = "%s"%(movie[x]))
        except:
            pass

    try:
        embed.add_field(name= "Episodes:", value = "%s"%(moviesDB.get_movie_episodes(movieID)["data"]["number of episodes"]))
    except:
        pass
    return [embed, movie]


def syntax(command):
    cmd_and_aliases = "|".join([str(command), *command.aliases])
    params = []

    for key, value in command.params.items():
        if key not in ("self", "ctx"):
            params.append(f"[{key}]" if "NoneType" in str(value) else f"<{key}>")

    params = " ".join(params)

    return f"```{cmd_and_aliases} {params}```"

def noEmbedSyntax(command):
    cmd_and_aliases = "|".join([str(command), *command.aliases])
    params = []

    for key, value in command.params.items():
        if key not in ("self", "ctx"):
            params.append(f"[{key}]" if "NoneType" in str(value) else f"<{key}>")

    params = " ".join(params)

    return f"{cmd_and_aliases} {params}"

def getPrefix(id):
    return DiscordGuild.find_one({"id":id}, {"prefix"})["prefix"]



async def ChoiceEmbed(self, ctx, choices:list, TitleOfEmbed:str, ReactionsList=['1️⃣', '2️⃣', '3️⃣', '4️⃣','5️⃣','6️⃣','7️⃣','8️⃣','9️⃣','🔟'],p:discord.Member=None,EmbedToEdit=None):
    count = 0
    reactionlist = []
    emptydict = {}
    finalstr = ""
    if len(choices)<=len(ReactionsList):
        for x in choices:
            emptydict[ReactionsList[count]]=x
            reactionlist.append(ReactionsList[count])
            finalstr+="%s %s\n"%(ReactionsList[count], x)
            count+=1
        embed = discord.Embed(title = TitleOfEmbed, description = finalstr, color = ctx.author.color)
        if EmbedToEdit!=None:
            EmbedToEdit = await EmbedToEdit.edit(embed=embed)
            EmbedToEdit.clear_reactions()
            for x in reactionlist:
                await EmbedToEdit.add_reaction(x)
        else:
            ThisMessage = await ctx.channel.send(embed=embed)
            for x in reactionlist:
                await ThisMessage.add_reaction(x)
        if not p:
            p=ctx.author

        def check(reaction, user):
            return user==p and str(reaction.emoji) in reactionlist and reaction.message == ThisMessage
        confirm = await self.client.wait_for('reaction_add',check=check, timeout = 60)
        try:
            if confirm:
                rawreaction = str(confirm[0])
                if EmbedToEdit!=None:
                    return[emptydict[rawreaction], EmbedToEdit]
                else:
                    return [emptydict[rawreaction], ThisMessage]
        except TimeoutError:
            await ctx.channel.send("You took too long! I guess we arent doing this.")
    else:
        chosen=False
        pgnum=1
        while chosen==False:
            if pgnum==1:
                choices=choices[0:9]
            if pgnum==2:
                choices=choices[10:len(choices)]
            for x in choices:
                emptydict[ReactionsList[count]]=x
                reactionlist.append(ReactionsList[count])
                finalstr+="%s %s\n"%(ReactionsList[count], x)
                count+=1
            embed = discord.Embed(title = TitleOfEmbed, description = finalstr, color = ctx.author.color)
            if EmbedToEdit!=None:
                EmbedToEdit = await EmbedToEdit.edit(embed=embed)
                EmbedToEdit.clear_reactions()
                for x in reactionlist:
                    await EmbedToEdit.add_reaction(x)
            else:
                ThisMessage = await ctx.channel.send(embed=embed)
                for x in reactionlist:
                    await ThisMessage.add_reaction(x)
                await ThisMessage.add_reaction("➡️")
                await ThisMessage.add_reaction("⬅️")

            if not p:
                p=ctx.author

            def check(reaction, user):
                return user==p and str(reaction.emoji) in reactionlist and reaction.message == ThisMessage
            confirm = await self.client.wait_for('reaction_add',check=check, timeout = 60)
            try:
                if confirm:
                    rawreaction = str(confirm[0])
                    if rawreaction=="➡️":
                        pgnum+=1
                        if pgnum>2:
                            pgnum=2
                    elif rawreaction=="⬅️":
                        pgnum-=1
                        if pgnum<1:
                            pgnum=1
                    else:
                        if EmbedToEdit!=None:
                            return[emptydict[rawreaction], EmbedToEdit]
                        else:
                            return [emptydict[rawreaction], ThisMessage]
            except TimeoutError:
                await ctx.channel.send("You took too long! I guess we arent doing this.")





async def AddChoices(self, ctx, choices:list, MessageToAddTo, p:discord.Member=None):
    for x in choices:
        await MessageToAddTo.add_reaction(x)
    if p==None:
        p=ctx.author
    def check(reaction, user):
        return user==p and str(reaction.emoji) in choices and reaction.message == MessageToAddTo
     
    confirm = await self.client.wait_for('reaction_add',check=check, timeout = 60)
    try:
        if confirm:
            print("Yes, This check worked")
            return str(confirm[0])
    except TimeoutError:
        await ctx.channel.send("You took too long!")
        return "Timeout"



class missingItem(commands.CommandError):
    def __init__(self, user, missingItem):
        self.user=user
        self.missingItem=missingItem

def hasItem(itemToCheckFor):
    def predicate(ctx):
        if itemToCheckFor.lower()=="pc":
            inv = mulah.find_one({"id":ctx.author.id}, {"inv"})["inv"]
            for x in inv:
                if "parts" in x.keys():
                    return True
            raise missingItem(ctx.author, itemToCheckFor)
        elif InvCheck(ctx.author,itemToCheckFor):
            return True
        else:
            raise missingItem(ctx.author, itemToCheckFor)
    return commands.check(predicate)



async def StoryEmbed(self, ctx, embedict:list):
    complete = False
    count = 0
    while complete == False:
        if count==len(embedict):
            complete = True
            break
        currentembed = embedict[count]
        embed = discord.Embed(title = currentembed["title"], description = currentembed["description"] ,color =ctx.author.color)
        try:
            if "file" in currentembed.keys():
                await editthis.edit(embed=embed, file = discord.File(currentembed["file"]))
            else:
                await editthis.edit(embed=embed)
        except:
            if "file" in currentembed.keys():
                editthis = await ctx.channel.send(embed=embed, file = discord.File(currentembed["file"]))
            else:
                editthis = await ctx.channel.send(embed=embed)
        await editthis.add_reaction("▶️")
        def check(reaction,userr):
            return userr==ctx.author and str(reaction.emoji)=="▶️" and reaction.message==editthis
        confirm = await self.client.wait_for('reaction_add', check=check, timeout = 60)
        try:
            if confirm:
                await editthis.clear_reactions()
                pass
                count+=1
        except asyncio.TimeoutError:
            await editthis.edit(embed=discord.Embed(title = "You took too long", color = ctx.author.color))


