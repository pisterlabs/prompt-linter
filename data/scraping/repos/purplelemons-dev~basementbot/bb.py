# The Basement Bot
# Documentation:    https://discordpy.readthedocs.io/en/latest/api.html
# Event reference:  https://discordpy.readthedocs.io/en/latest/api.html?highlight=role%20mention#event-reference

from timeit import default_timer as dt

st=dt() # start of all code
# makes startup prettier. dont worry about it this totally doesnt hinder startup performance.
print()

#import warnings
#warnings.filterwarnings("ignore")

# discord-oriented tools
import discord
from discord.ext import commands
import json
import asyncio as aio
from discord_slash import SlashContext, SlashCommand
from discord_slash.utils.manage_commands import create_option, create_choice, create_permission
# system
import sys
from os.path import isfile, dirname, getsize, getmtime
from contextlib import redirect_stdout
from dotenv import dotenv_values as dev
import os
import psutil
import threading
from traceback import print_exc
# time
from datetime import datetime as datet, timedelta
from pytz import timezone
import time
# misc.
import random as rand
from hashlib import sha256 as sha
from io import StringIO
import http.server as httpserver
import openai
#from profanity_filter import ProfanityFilter
#from profanity_check import predict as pf_predict
# add something about pip install discord.py[voice]

BOOT_TIME=datet.now().strftime("%m/%d/%Y %H:%M:%S")


# initialize discord API
client=commands.Bot(
    command_prefix="",
    intents=discord.Intents.all(),
    max_messages=1024,
    owner_id=483000308876967937 # Me
)
slash=SlashCommand(client, sync_commands=True)
token=dict(dev(".env"))["bottoken"]
openai.api_key=dict(dev(".env"))["OPENAI_API_KEY"]
NUKE_PASS=dict(dev(".env"))["nukePass"]
MONTHS={
    1:'January',
    2:'February',
    3:'March',
    4:'April',
    5:'May',
    6:'June',
    7:'July',
    8:'August',
    9:'September',
    10:'October',
    11:'November',
    12:'December'
}

# profanity library init
#profanity_filter=ProfanityFilter()
#profanity_filter.censor_char="."

# current directory
client.dir=dirname(__file__)

GUILDS=[858065227722522644]
STAFF_ID=936734394591363182
DEPRECATED=[
    "setraw",
    "jd",
    "log",
    "test",
    "retrieve",
    "announce",
    "poll",
    "create",
    "stats",
    "snow",
    "msgs",
    "del",
    "warn",
    "recent",
    "count",
    "dm",
    "sha"
]

ALIASES=DEPRECATED+[
    "clear",
    "snowflake",
    "statistics",
    "hash"
]

# user defined functions
def create(data:dict[int,dict[str,]], id_:int, name:str, time:str): # creates a member in the member data dictionary
    if id_ in data:
        return

    data[id_]={
        "name":f"{name}",
        "isMuted":False,
        "warnCount":0,
        "muteCount":0,
        "banCount":0,
        "msgCount":0,
        "permLvl":0,
        "countHigh":0,
        "countFails":0,
        "joinDate":time,
        "birthday":"",
        "history":[]
    }

def save(dictionary): # saves member data in dictionary to hard storage
    direct=dirname(__file__)
    with open(f"{direct}/bb.json", "w") as f: json.dump(dictionary, f)

def load(): # loads member data
    with open(f"{client.dir}/bb.json", "r") as f: r=json.loads(f.read())
    new={int(i):r[i] for i in r if i.isdigit()}
    for i in r:
        if not i.isdigit(): new[i]=r[i]
    return new

def isListed(m,list_): return m.author in list_

def toDigits(num:str) -> list[int]:
    """
    Return a list of base-36 minus 10 converted string.
    For #alphabet-counting
    """
    return [-1]+[int(i,36)-10 for i in num]

def isNext(current:str,nextNum:str) -> bool:
    current, nextNum = current.lower(), nextNum.lower()
    if current==nextNum=="a":
        return True

    elif current[-1]=="z" and nextNum[-1]=="a":
        digcurrent, dignextNum=toDigits(current),toDigits(nextNum)
        return digcurrent[:-2]==dignextNum[:-2] and digcurrent[-2]+1==dignextNum[-2] or (current=="z"*len(current) and nextNum=="a"*len(nextNum) and len(current)==len(nextNum)-1)

    elif len(current)==len(nextNum):
        digcurrent, dignextNum=toDigits(current),toDigits(nextNum)
        return digcurrent[:-1]==dignextNum[:-1] and digcurrent[-1]+1==dignextNum[-1]

    else:
        return False

async def nickToMember(nick:str,author:discord.Member,ctx:SlashContext):
    memberGuesses:list[discord.Member]=[kiddie for kiddie in client.basement.members if kiddie.display_name.startswith(nick) or kiddie.name.startswith(nick)]
    if len(memberGuesses)==1:
        return memberGuesses[0]
    else:
        glist="\n"+"\n".join(kiddie.display_name for kiddie in memberGuesses)
        await ctx.send(f"{author.mention}, there are `{len(memberGuesses)}` people that match the username you provided me with (\"{nick}\") please check your spelling and/or specify further.{glist}")
        return

def nickToMemberSync(nick:str)->list[discord.Member]:
    return [kiddie for kiddie in client.basement.members if kiddie.display_name.startswith(nick) or kiddie.name.startswith(nick)]

def verify(i:str)->bool: return i.strip() and not i.strip().startswith("#") and len(i.strip())-1

def getPermissions()->dict[str,str]:
    with open(f"{client.dir}/bb.py",'r') as f: fread=[i.strip() for i in f.read().split("class slashCmds:")[-1].split('\n')]
    return {l[len("async def _"):].split("(")[0]:fread[i+1][1] for i,l in enumerate(fread) if l.startswith("async def _") and fread[i+1][1] in ["a","s","e"]}

def checkPermission(cmdPerm:str,memberPerm:str)->bool:
    translation={
        "a":2,
        "s":1,
        "e":0
    }
    return translation[cmdPerm]<=translation[memberPerm]

def parseDesciptions(cmd:str=None,maxPerm:str="e")->dict:
    "this was a ***** to code"
    print(maxPerm)
    # convert non-listed cmd to all cmds
    permissions=getPermissions()
    if not cmd: cmds=["/"+cmd[1:] for cmd in dir(slashCmds) if not cmd.startswith("__") and checkPermission(permissions[cmd[1:]],maxPerm)]
    else: cmds=[cmd]
    out={}

    # grabs file from "class slashCmds:" down as list `fread`, no whitespaces
    with open(f"{client.dir}/bb.py",'r') as f: fread=[i.strip() for i in f.read().split("class slashCmds:")[-1].split('\n') if verify(i)]

    for command in cmds:
        for idx,line in enumerate(fread):
            if line.startswith("name=") and fread[idx-1]=="@slash.slash(" and line[6:-2]==command[1:]:
                name=command[1:]
                if fread[idx+1].startswith("description=f\""): desc=fread[idx+1][14:-2]
                else: desc=fread[idx+1][13:-2]
                out[name]=desc
    return out

def convertAttr(attr:str)->str:
    translations={
        "name":"Name",
        "isMuted":"Is muted?",
        "warnCount":"Warn count",
        "muteCount":"Mute count",
        "banCount":"Ban count",
        "msgCount":"Message count",
        "permLvl":"Permission level",
        "countHigh":"Highest count",
        "countFails":"Count fails",
        "joinDate":"Date joined",
        "birthday":"Birthday"
    }
    return translations[attr] if attr in translations else "INTERNALERROR"

def scanMessage(phrase:str, string:str) -> bool:
    phrase=phrase.lower()
    string=string.lower()
    chars=['"',"'",".",",","-","‎","*"]
    try: percent=len([i for i in string if i==" "])/len(string)
    except ZeroDivisionError: percent=1
    for char in chars:
        if string!="he'll" and phrase in string.replace(char,"").split(" "):# and not phrase in string:
            return string.replace(char,"").replace(phrase, f"||{phrase}||")
    if phrase in string.replace(" ","") and not phrase in string and percent>.42:
        return string.replace(" ","").replace(phrase, f"||{phrase}||")
    elif f" {phrase} " in string or phrase==string:
        return string.replace(phrase, f"||{phrase}||")
    else:
        return False

def properOrdinal(number:str)->str:
    if number.endswith("0") or number.endswith("4") or number.endswith("5") or number.endswith("6") or number.endswith("7") or number.endswith("8") or number.endswith("9"):
        return number+"th"
    elif number[1]=="1":
        return number+"th"
    elif number.endswith("1"):
        return number+"st"
    elif number.endswith("2"):
        return number+"nd"
    elif number.endswith("3"):
        return number+"rd"
    else: raise ValueError("AAAAAAAAAAAAAAAAAAAAAAAA")

def correctBday(bday:str)->str:
    bday=bday.replace("/0","/")
    return bday[1:] if bday.startswith("0") else bday

def properBday(bday:str)->str:
    if bday.startswith("0") or "0/" in bday: bday=correctBday(bday)
    month,day=bday.split("/")
    return f"{MONTHS[int(month)]} {properOrdinal(day)}"

def removePunctuation(words:list[str])->list[str]:
    out=[]
    for word in words:
        for i in (".","!","?","'",",",'"'):
            if i in word: word=word.replace(i,"")
        out.append(word)
    return out

# Grab our json/dictionary
client.Data=load()
if client.Data:
    print("[BasementBot] Client data read sucessfully!")

####################################################################################################

class Interbot (httpserver.BaseHTTPRequestHandler):

    def end_headers(self):
        #now=datet.now().strftime("%H:%M:%S")
        #print(f"[Interbot] {now} - triggered end_headers()")
        self.send_header("Access-Control-Allow-Origin","https://bot.thebasement.group")
        super().end_headers()

    # Silence requests log
    def log_request(self, code="-", size="-"):
        return

    def do_OPTIONS(self):
        if self.path=="/cmd":
            self.send_response(200)
            self.send_header("Access-Control-Allow-Headers","content-type, accepts")
            self.send_header("Access-Control-Allow-Methods","POST,OPTIONS")
            self.send_header("Content-type","text/plain")
            self.send_header("allow","OPTIONS, POST")
            self.end_headers()
        else:
            self.quick_code(404)

    def do_GET(self):
        try:
            self.send_response(200)
            self.send_header("Content-type","text/plain")
            self.end_headers()
            self.wfile.write("Hey there!\n".encode())
        except:
            print_exc()
    
    def body(self)->dict[str,]:
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length"))))

    def validate(self,password:str)->bool:
        with open("/root/basementbot/passwords",'r') as f: return password in [i.strip() for i in f.readlines()]

    # 2xx responses. adds headers and sends response
    def _200(self,data:dict[str,]):
        self.send_response(200)
        self.send_header("Content-type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    # any HTTP response code. adds headers and sends response
    def quick_code(self,code:int,message:str=None):
        self.send_response(code)
        if not message: self.end_headers()
        else:
            self.send_header("Content-type","application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"msg":message}).encode())
    
    def parse_args(self,arg:str):
        "Auto converts a `;`-seperated string of keys into a list of integers and strings."
        if type(arg)==str:
            arg=arg.strip()
            return int(arg) if arg.isdigit() else arg
        return arg

    def do_POST(self):
        if not self.path in ("/test","/cmd"):
            self.quick_code(404)
            return
        body=self.body()
        
        try: isValid=self.validate(body["password"])
        except: self.quick_code(400,"Expected: \"password\"")

        else:
            if isValid:
                if self.path=="/test":
                    self.send_response_only(200,"all good!")
                    self._200({
                        "msg":"hi"
                    })
                    return
                elif self.path=="/cmd":
                    # cmd can be a variety of commands
                    #  dgn - diagnostics
                    #  ret - gets the value of some item in the dictionary
                    #  set - changes the value of some item in the dictionary
                    #  tst - experimental
                    #  bbt - execute slash commands
                    #  res - will resolve discord IDs to json objects that note it's attributes
                    #  lok - will lookup a member's nickname or username based on .startswith()
                    #  rrc - reaction roles configuration
                    #  qev - query events
                    #  rmg - edit reaction role message in the reaction roles channel
                    if "command" in body:
                        command:str=body["command"]
                        if command=="dgn":
                            # args:     None
                            # returns:  a dict of current operating values
                            data={
                                "upt":BOOT_TIME,
                                "cpu":str(psutil.cpu_count()),
                                "cpr":str(psutil.cpu_percent()),
                                "ram":str(psutil.virtual_memory().available),
                                "rpr":str(psutil.virtual_memory().percent),
                                "pid":str(os.getpid())
                            }
                            self._200(data)
                            #print(f"[Interbot] i just sent {data}")
                        elif command=="set":
                            # args:     path,value
                            #  path:    required. a list of keys
                            #  value:   required. a string of the new value of path
                            # returns:  a confirmation, client.Data is modified
                            if not "path" in body:
                                self.quick_code(400,"Expected: \"path\"")
                                return
                            elif not "value" in body:
                                self.quick_code(400,"Expected: \"value\"")
                                return
                            path=body["path"]
                            value=self.parse_args(body["value"])

                            for place in path:
                                # will (hopefully) prevent rexec
                                if "]" in place or ";" in place: # set a,b,c target
                                                                 # set a,b]; kmksdo[fmiko[g]] #
                                    self.quick_code(418,"no")
                                    return
                            if type(value)==str:
                                if ";" in value:
                                    # will (hopefully) prevent rexec
                                    self.quick_code(418,"no")
                                    return

                            try:
                                execCommand="client.Data"+"".join(f"[\"{arg}\"]" if type(arg)==str else f"[{arg}]" for arg in path)
                                execCommand+="="+(f"\"{value}\"" if type(value)==str else str(value))
                                # this isn't dangerous at all
                                exec(execCommand)
                                target=client.Data
                                for place in path:
                                    target=target[place]
                                if str(target)!=str(target).replace('"',"'"):
                                    print(f"target: {target} - {type(target)}")
                                    print(f"path: {path}")
                                    print(f"command: {execCommand}")
                                    raise ValueError
                            except ValueError:
                                print_exc()
                                self.quick_code(500,"Unsuccessfully set.")
                                return
                            except:
                                print_exc()
                                self.quick_code(500,"Internal error\nContact josh@thebasement.group")
                                return

                                "⍼"
                            else:
                                self._200({
                                    "msg":"Success!"
                                })
                                return
                        elif command=="ret":
                            # args:     path
                            #  path:    required. the path to the desired target
                            # returns:  the bot's dictionary record at the target
                            if not "path" in body:
                                self.quick_code(400,"Expected: \"path\"")
                                return

                            path=body["path"]
                            target=client.Data
                            for place in path:
                                place=place.strip()
                                if place.isdigit(): place=int(place)
                                try:
                                    target=target[place]
                                except KeyError:
                                    self.quick_code(404,f"{place} not found")
                                    return
                            else:
                                # Change all integer values in target to strings
                                if type(target)==dict:
                                    for key,value in target.items():
                                        if type(value)==int: target[key]=str(value)
                                self._200(target)
                        elif command=="res":
                            # args:     id
                            #  id:      required. the id of the unknown object
                            # returns:  a dictionary of a discord object
                            basem:discord.Guild=client.basement
                            converters=(basem.get_channel,basem.get_member,basem.get_role,client.get_emoji)
                            for converter in converters:
                                conv=converter(int(body["id"]))
                                try:
                                    if conv is not None:
                                        self._200(conv.__dict__())
                                        return
                                except AttributeError:
                                    print_exc()
                                    self.quick_code(500,f"Error resolving {body['id']}")
                            else:
                                self.quick_code(404,"Could not resolve")
                        elif command=="lok":
                            # args:     nick
                            #  nick:    a string representing the nickname
                            # returns:  a member object
                            if not "nick" in body:
                                self.quick_code(400,f"Expected: \"nick\"")
                                return
                            nick:str=body["nick"]
                            self._200({
                                member.display_name:client.basement.get_member(member.id).__dict__() for member in nickToMemberSync(nick)
                            })
                        elif command=="rrc":
                            # args:     roles
                            #  roles:   a dictionary of emoji to role names
                            # returns:  a status message
                            if not "roles" in body:
                                self.quick_code(400,"Expected: \"roles\"")
                                return
                            roles:dict[str,str]=body["roles"]
                            newTable:dict[str,str]={}
                            for emoji,role in roles.items():
                                for gRole in client.basement.roles:
                                    if gRole.name.lower().startswith(role.lower()):
                                        newTable[emoji]=str(gRole.id)
                                        break
                                else:
                                    self.quick_code(404,f"Could not find role {role}")
                                    return
                            client.Data["RR"]["table"]=newTable

                            async def _rrc():
                                messages:list[discord.Message]=await client.get_channel(client.Data["RR"]["channel"]).history(limit=1).flatten()
                                message=messages[0]
                                for reaction in message.reactions:
                                    if reaction.emoji not in client.Data["RR"]["table"]:
                                        print(f"Clearing {reaction.emoji} ...")
                                        await reaction.clear()

                            aio.run_coroutine_threadsafe(_rrc(),client.loop)
                            self._200({
                                "msg":"Success"
                            })
                        elif command=="qev":
                            # args:     event
                            pass
                        elif command=="rmg":
                            # args:     none
                            # returns:  a status message
                            try:
                                message="React with any of the emojis listed below to get a role! (and be honest, we know where you live)\n"
                                message+="\n".join(f"    {emoji} - {client.basement.get_role(int(role)).mention}" for emoji,role in client.Data["RR"]["table"].items())
                                async def _rmg():
                                    channel:discord.TextChannel=client.basement.get_channel(client.Data["RR"]["channel"])
                                    lastmsg:list[discord.Message]=await channel.history(limit=1).flatten()
                                    if lastmsg[0].author!=client.user:
                                        await lastmsg[0].delete()
                                        await channel.send(message)
                                    else: await lastmsg[0].edit(content=message)
                                aio.run_coroutine_threadsafe(_rmg(),client.loop)
                            except:
                                now=datet.now().strftime("%m/%d/%Y %H:%M:%S")
                                print(f"{now} [Interbot] Error in rmg")
                                print_exc()
                                self.quick_code(500,"Internal error\nContact")
                            else:
                                self._200({
                                    "msg":"Success"
                                })
                        else:
                            self.quick_code(404,f"Unknown command: \"{command}\"")

                    else:
                        self.quick_code(400,"Expected: \"command\"")
                    return
            else:
                self.quick_code(401,"Incorrect password")
                return

####################################################################################################

@client.event
async def on_ready(): # do all this on startup...
    st=dt()
    client.basement=client.get_guild(858065227722522644) # basement guild
    #client.botLog=client.basement.get_channel(862004853763866624) # bot log channel
    #client.modLog=client.basement.get_channel(888981385585524766) # mod log channel
    client.counting=client.basement.get_channel(903043416571662356) # counting channel
    client.welcome=client.basement.get_channel(858065227722522647) # welcome channel
    #client.checklist=client.basement.get_channel(862567884286984202)
    client.rules=client.basement.get_channel(858156716238438421)
    #client.group=client.basement.get_channel(871455541602430986) - channel deleted
    client.faq=client.basement.get_channel(858200405933555722)
    client.basementC=client.basement.get_channel(858157764530798612)
    client.wRole=client.basement.get_role(908491255812616252) # welcome role
    client.kRole=client.basement.get_role(858234453653848065) # kiddie role
    client.modRole=client.basement.get_role(858416941185761290) # moderator role
    client.adminRole=client.basement.get_role(858223675949842433) # admin role
    client.breakbbC=client.basement.get_channel(910345635365015613) # break-basementbot channel
    client.wrdAssocC=client.basement.get_channel(859807366219694080) # word association channel
    client.lSSC=client.basement.get_channel(933183879983013968) # newest minigame, long-story-short
    client.testC=client.basement.get_channel(933576021641400370) # test channel in dev category
    client.alphaCountC=client.basement.get_channel(932125853003943956) # alphabet counting channel, minigame
    client.announcementC=client.basement.get_channel(858156757788524554)
    client.spamC=client.basement.get_channel(942576682395660358) # spam channel
    client.botInteractC=client.basement.get_channel(858422701165510690) # bot-interactions channel
    client.noahb=client.basement.get_member(806307221171994624) # noah
    client.josh=client.basement.get_member(483000308876967937) # josh smith
    client.modGenC=client.basement.get_channel(862035189159690280) # mod-general channel
    client.staffRole=client.basement.get_role(936734394591363182) # staff role
    client.alertC=client.basement.get_channel(960376643329851432) # bot alerts channel
    client.disciplineC=client.basement.get_channel(973750522651766794) # discipline channel
    client.deletedMessages=[] # for tracking which messages we've deleted
    client.lastSave=dt() # used for autosaving
    client.localTZ=timezone("US/Central")
    print(f"Set up client variables in {dt()-st}s!")

    # edit reconnect message, if it exists
    try: recChannelID, recMsgID=client.Data["reconnect"]
    except KeyError: pass
    except Exception as e: print(f"{_dt} [BasementBot] There was an internal error during reconnect message edit.\n{repr(e)}")
    else:
        channel:discord.TextChannel=client.get_channel(recChannelID)
        message:discord.PartialMessage=channel.get_partial_message(recMsgID)
        try:
            await message.edit(embed=discord.Embed(title="Reconnected!",color=discord.Color.green()))
        except:
            print("no reconnect message")

    _dt=datet.now().strftime("%m/%d/%Y %H:%M:%S") # current time

    # repair files
    for channel in client.basement.channels:
        if not isfile(f"{client.dir}/logs/{channel.id}.txt"):
            with open(f"{client.dir}/logs/{channel.id}.txt", "w", encoding='utf-8') as f: print((f"BEGINNING OF #{channel.name}\n"), file=f) # if the channel log does not exist, create it

    for kiddie in [member for member in client.basement.members if not member.bot]: # in the case that data was corrupted, repair kiddie files
        kiddie:discord.Member
        if kiddie.id not in client.Data:
            create(client.Data, kiddie.id, kiddie.display_name, _dt)
            for role in kiddie.roles: # determine permission level for every kiddie
                if role.name=="Administrator":
                    client.Data[kiddie.id]["permLvl"]=2
                    break
                elif role.name=="Moderator":
                    client.Data[kiddie.id]["permLvl"]=1
                    break
            if kiddie.id==806307221171994624: client.Data[kiddie.id]["permLvl"]=3 # noah

        elif "joinDate" not in client.Data[kiddie.id]:
            print(f"[BasementBot] Created joinDate entry for \"{kiddie.display_name}\"")
            joinDate:datet=kiddie.joined_at
            client.Data[kiddie.id]["joinDate"]=joinDate.astimezone(client.localTZ).strftime("%m/%d/%Y %H:%M:%S")
        
        # fix this later
        #if client.adminRole not in kiddie.roles:
        #    for char in kiddie.display_name:
        #        if char.lower() not in [i for i in "abcdefghijklmnopqrstuvwxyz()!~0123456789[]<>{}-_+=*^%$#@&/\\|?.,;:'"]:
        #            try:
        #                await kiddie.send(f"Dear {kiddie.mention},\n\nI noticed you had special characters in your name, and this messes with my systems. Will you kindly change it to something without special characters while Josh tries to figure out a solution? If you didn't use a special character, please harrass MR_H3ADSH0T#0001 (Josh) until he fixes it lol\n\n\t\tSincerely, The Basement Staff Team")
        #            except:
        #                await client.botAlertsC.send(embed=discord.Embed(title="",description=f"{kiddie.mention} has special characters in their name, and I couldn't DM them to fix it. Please change their name to something without special characters.",color=discord.Color.red()))
        #            break

    # status
    notBots=[member for member in client.basement.members if not member.bot] # i don't know why, but the bot is always one less than the current count...
    await client.change_presence(activity=discord.Streaming(name=f"we've got {len(notBots)} members!",url="https://thebasement.group"))

    # sucess report
    await client.alertC.send(embed=discord.Embed(title="Connected",description=f"Bot connected at `{_dt}`",color=discord.Color.green())) # report connection
    hour, minute = client.Data["randTime"]
    print(f"[BasementBot] Sucessfully loaded bot configurations! Startup took {dt()-st}s!\nGood-morning Time: {hour}:{minute}")
    print(f"Boot time: {BOOT_TIME}")

    # main loop
    while True:
        # dont DoS our bot, keep it slow but not too slow
        await aio.sleep(1)
        
        #if not server.isRunning:
        #    server.server_close()
        #    print("[Interbot] ok i shut down you dumb f***")
        now=datet.now()
        ##################################
        ######### ROUTINE CASUAL #########
        ##################################
        # handles good morning
        if client.Data["GMday"]!=now.day and [now.hour,now.minute]==client.Data["randTime"]: # at a random time in the morning
            await client.basementC.send(f"Good morning guys!")
            client.Data["GMday"]=now.day # reset day
            client.Data["randTime"]=[rand.randint(5,14),rand.randint(0,59)] # new random time
        # handles happy birthday
        if (now.hour,now.minute,now.second)==(9,0,0): # 9am
            monthDay=now.strftime("%m/%d")
            for member in [i for i in client.Data if type(i)==int]:
                try:
                    if client.Data[member]["birthday"]==monthDay:
                        member:discord.Member=client.basement.get_member(member)
                        await client.basementC.send(embed=discord.Embed(title="HAPPY BIRTHDAY!",description=f"Wishing **{member.display_name}** a happy birthday! :tada:",color=member.color))
                except KeyError:
                    await client.alertC.send(f"{client.josh.mention} i had an error finding {member.display_name}'s birthday data")
                    print(f"Coudn't find 'happy birthday' for {member} ({client.Data[member]['name']})")
        # handles good night
        if (now.hour,now.minute,now.second)==(23,59,59) and (now.month,now.day)!=(12,31): # will say "Goodnight, y'all" at 11:59:59pm EXCEPT on new year's
            await client.basementC.send(f"Good night, y'all 😴")

        ##################################
        ####### ROUTINE MODERATION #######
        ##################################
        # executes "/log" every hour
        #if (now.minute,now.second)==(0,0):
        #    online,o,dt_str="",0,now.strftime("%m/%d/%Y %H:%M:%S")
        #    for kiddie in client.basement.members:
        #        if kiddie.raw_status!="offline" and kiddie.bot==False:
        #            online+=" - "+kiddie.display_name+"\n"
        #            o+=1
        #    await client.alertC.send(f"This was an auto-initiated process. Use `/log` to see this list.\n{o} online members at `{dt_str}`:```\n{online}```") # send to #modlogging
        #    await aio.sleep(10)
        # handles bi-hourly saves
        if now.second==0 and not now.minute%30: save(client.Data)
        # handles spam purge
        if (now.hour,now.minute,now.second)==(0,0,0):
            deleteTime:datet=datet(
                year=now.year,
                month=now.month,
                day=now.day,
                hour=now.hour,
                minute=now.minute,
                second=now.second
            ) - timedelta(days=1)

            spamC:discord.TextChannel=client.spamC
            await spamC.purge(limit=8192,before=deleteTime)
            #await client.modLog.send(f"```{_dt} purged #spam messages.```")

        #################################
        ######## CALENDAR EVENTS ########
        #################################
        # handles happy new year
        if (now.month,now.day,now.hour,now.minute)==(12,31,23,59):
            if now.second==0:
                await client.basementC.send("Get ready for the new year countdown!")
            cd=""
            if now.second>10:
                cd=f"***{60-now.second}***"
            else:
                cd=f"{60-now.second}"
            await client.basementC.send(cd)
        # handles basement anniversary
        if (now.month,now.day,now.hour,now.minute,now.second)==(6,25,19,24,52):
            happyAnniv=discord.Embed(description=f"Happy Birthday, Basement! Here's to another year of awesomeness!\nOh, and the next time you see {client.noahb.mention}, tell him how much you like The Basement! (He'll blush haha)",color=discord.Color.purple())
            happyAnniv.set_author(name="HAPPY BASEMENT ANNIVERSARY!!!",icon_url="https://test.thebasement.group/images/tada.png")
            await client.basementC.send(embed=happyAnniv)
        # handles valentine's day
        if (now.month,now.day,now.hour,now.minute,now.second)==(2,14,12,1,0):
            valintDay=discord.Embed(description="Have a great valintines day, whether you're a single pringle, or a hopeless romantic! Just know that someone out there loves you very very much ❤️",color=discord.Color.red())
            valintDay.set_author(name="Happy Valentine's Day!",icon_url="https://test.thebasement.group/images/heart.png")
            await client.basementC.send(embed=valintDay)
        # handles 2/22/22 22:22:22
        if (now.year-2000,now.month,now.day,now.hour,now.minute,now.second)==(22,2,22,22,22,22):
            _dtstr=now.strftime("%m/%d/%Y %H:%M:%S")
            await client.basementC.send(embed=discord.Embed(title="Happy twosday!",description=f"The time is:\n```{_dtstr} CST```"))

@client.event
async def on_member_join(member:discord.Member):

    _dt=datet.now().strftime("%m/%d/%Y %H:%M:%S") # current time

    if member.id!=864881059187130458 and member.id not in client.Data: # not Josh S. (alt)
        await client.welcome.send(f"{client.wRole.mention}, give a warm welcome to our newest member, {member.display_name}!") # public welcome

    elif member.id in client.Data:
        await client.welcome.send(f"{member.display_name} has returned!")

    if member.id==864881059187130458: # be funny bc alt
        await client.welcome.send(f"Yo <@864881059187130458> is here. Josh must be testing stuff again :man_shrugging:")

    # private DM
    await member.send(f"Dear, {member.mention}, thank you for joining **{client.basement.name}**! Please read the {client.rules.mention} channel before exploring the server. Also, the Admin team loves to hear everyone's brilliant ideas, so using {client.faq.mention} is encouraged!\n\n        Sincerely, the Admin team")

    # give member childrne roll
    await member.add_roles(client.kRole)

    create(client.Data, member.id, member.display_name, _dt)
    save(client.Data)

    # update member count
    notBots=[i for i in client.basement.members if not i.bot] # see on_ready()
    await client.alertC.send(embed=discord.Embed(title="Member Joined",description=f"**{member.display_name}** just joined, bringing the member count to: `{len(notBots)}`",color=discord.Color.blue()))
    await client.change_presence(activity=discord.Streaming(name=f"we've got {len(notBots)} members!",url="https://thebasement.group"))

@client.event
async def on_member_remove(member:discord.Member):
    await client.welcome.send(f"{member.display_name} has left us... wishing them a happy life :)")
    notBots=[i for i in client.basement.members if not i.bot] # see on_ready()

    await client.alertC.send(embed=discord.Embed(title="Member Left",description=f"**{member.name}#{member.discriminator}** has left the server.\nThere are now {len(notBots)} members in the server.",color=discord.Color.black())) # had to modify the discord color file for this to work to work with the new embeds
    await client.change_presence(activity=discord.Streaming(name=f"we've got {len(notBots)} members!",url="https://thebasement.group"))

@client.event
async def on_member_update(before:discord.Member, after:discord.Member):
    now=datet.now()
    _dt=now.strftime("%m/%d/%Y %H:%M:%S") # get current time/date

    with open(f"{client.dir}/logs/nicknames.txt",'a', encoding="utf-8") as f:
        if before.display_name!=after.display_name:
            print(f"{_dt} {before.name} \"{before.display_name}\" -> \"{after.display_name}\"")

            for word in client.Data["blw"]:
                if scanMessage(word,after.display_name):
                    await after.edit(nick=before.display_name)
                    break

            # fix this later
            ## check and prevent special chars in everyone but admins
            #if client.adminRole not in before.roles:
            #    for char in after.display_name:
            #        if char.lower() not in [i for i in "abcdefghijklmnopqrstuvwxyz()!~0123456789[]<>{}-_+=*^%$#@&/\\|?.,;:'"]:
            #            await after.edit(nick=before.display_name)
            #            await after.send(f"Dear {after.mention},\n\nPlease don't use special characters in your nickname. If you didn't use a special character, please harrass MR_H3ADSH0T#0001 (Josh) until he fixes it lol\n\n\t\tSincerely, The Basement Staff Team")
            #            break

    with open(f"{client.dir}/logs/master.txt", "a", encoding="utf-8") as f:

        if before.status!=after.status and before.id!=691383407548301432: # change in status -- also dont include impersonator! creates clutter
            print(f"{_dt} {before.name} ({before.display_name})'s status was changed from \"{before.status}\" to \"{after.status}\"!", file=f)

        elif before.activity!=after.activity: # change in activity.... this is actually kinda creepy and a little bit intrusive
            print(f"{_dt} {before.name} ({before.display_name})'s activity was changed from {before.activity} to {after.activity}!", file=f)

        elif before.roles!=after.roles: # this is useful for identifying if that person was promoted or demoted
            # find out which role was changed
            bRoles,aRoles=[role.name for role in before.roles],[role.name for role in after.roles] # list of role names of before and after
            if len(bRoles)>len(aRoles): # find out which role was removed
                missing=str(list(set(bRoles)-set(aRoles))[0])
                print(f"{_dt} {before.name} ({before.display_name}) lost the role {missing}!", file=f)
                if missing in ["Admin","Mod"]: client.Data[before.id]["permLvl"]-=1 # if the role was important, subtract one... this is really quite clever actually
            if len(bRoles)<len(aRoles): # find out which role was added
                gained=str(list(set(aRoles)-set(bRoles))[0])
                print(f"{_dt} {before.name} ({before.display_name}) recieved the role {gained}!", file=f)
                if gained in ["Admin","Mod"]: client.Data[before.id]["permLvl"]+=1 # see line 113 comment

        elif before.pending!=after.pending: # no use for verification right now...
            print(f"{_dt} {before.name} ({before.display_name})'s verification was changed from {before.pending} to {after.pending}!", file=f)

@client.event
async def on_message_delete(message:discord.Message):
    _dt=datet.now().strftime("%m/%d/%Y %H:%M:%S") # get current time/date
    auth=message.author # author

    if message.author.id==client.user.id or message.channel.category_id in (872720713742704671,858162628537745408): pass # dont log our own messages

    if message.id in client.deletedMessages: # if the message was already deleted
            client.deletedMessages.remove(message.id) # remove it from the list
            return # do nothing

    elif message.attachments and message.channel not in (client.counting, client.alphaCountC, client.alertC): # if there are attachments
        for att in message.attachments:
            content="!" if not message.content else f": \"{message.content}\"" # get content if there is any
            await client.alertC.send(embed=discord.Embed(title="Message deleted!",description=f"**{auth.display_name}**'s message in {message.channel.mention} was deleted{content}\nURL: [{att.filename}]({att.proxy_url})",color=discord.Color.red()))

    elif message.channel not in (client.counting, client.alphaCountC, client.alertC): # if there are no attachments
        await client.alertC.send(embed=discord.Embed(title="Message deleted!",description=f"**{auth.display_name}**'s message in {message.channel.mention} was deleted: \"{message.content}\"",color=discord.Color.red()))

    if message.channel.category_id not in (872720713742704671,858162628537745408):
        with open(f"{client.dir}/logs/{message.channel.id}.txt", "a", encoding="utf-8") as f:
            try:
                print(f"{_dt} {auth.name} ({auth.display_name})'s message was deleted \"{message.clean_content}\" in {message.channel.name}", file=f)
            except AttributeError:
                print(f"{_dt} {auth.name} ({auth.display_name})'s message was deleted \"{message.clean_content}\" in (probably) the DMs", file=f)

    if message.channel==client.counting and message.content.isdigit() and not message.content.startswith("0"): # counting alert
        await client.counting.send(f"{auth.name}'s message \"`{message.clean_content}`\" was deleted! The current number is `{client.Data['counting']}`")

@client.event
async def on_message_edit(before:discord.Message,after:discord.Message):
    _dt=datet.now().strftime("%m/%d/%Y %H:%M:%S") # get current time/date
    auth=before.author # author
    Bclean=before.clean_content # before
    Aclean=after.clean_content # after

    if before.author.id in (client.user.id, 282859044593598464): # dont log our own messages or ProBot's messages
        pass

    elif before.content!=after.content:
        await client.alertC.send(embed=discord.Embed(title="Message edited!",description=f"**{auth.display_name}** edited their message in {before.channel.mention} from: \"{Bclean}\"\nTo: \"{Aclean}\"",color=discord.Color.orange()))

        if after.content.replace(":","")=="420":
            for reaction in after.reactions:
                # check if thumbs up
                if reaction.emoji=="\U0001f44d":
                    await reaction.clear()

    with open(f"{client.dir}/logs/{before.channel.id}.txt", "a",encoding="utf-8") as f:
        print(f"{_dt} {auth.name} ({auth.display_name}) changed their message from \"{Bclean}\" to \"{Aclean}\" (path: {before.channel.id}/{before.id})",file=f)

    if before.channel==client.counting and Bclean.isdigit(): # counting alert
        await before.reply(f"{auth.name} changed their message \"`{Bclean}`\")! The current number is `{client.Data['counting']}`",mention_author=False)

    if before.channel==client.alphaCountC and not before.content.startswith("> "):
        await after.delete()

    # Profanity filter
    for word in client.Data["blw"]:
        filtered=scanMessage(word,Aclean)
        if filtered:
            await after.delete()
            embed=discord.Embed(description=f"{auth.mention}'s message was edited to and deleted in {after.channel.mention}.\n\"{filtered}\"",color=discord.Color.red())
            await client.alertC.send(embed=embed)

@client.event
async def on_raw_reaction_add(payload:discord.RawReactionActionEvent):
    emoji:discord.partial_emoji.PartialEmoji=payload.emoji
    emojiAuth:discord.Member=client.basement.get_member(payload.user_id)
    try: channel:discord.TextChannel=await client.fetch_channel(payload.channel_id)
    except: return
    message:discord.Message=await channel.fetch_message(payload.message_id)

    if payload.channel_id==858156662703128577: # polls channel
        upEmoji:discord.Emoji=client.get_emoji(858556326548340746)
        downEmoji:discord.Emoji=client.get_emoji(858556294743064576)
        if not emoji in (upEmoji,downEmoji) and message.author==client.user:
            await message.remove_reaction(emoji,emojiAuth)
            return
        try:
            ups:discord.Reaction=list(filter(lambda x: x.emoji==upEmoji, message.reactions))[0]
            downs:discord.Reaction=list(filter(lambda x: x.emoji==downEmoji, message.reactions))[0]
        except IndexError: pass # this will happen if we cant find the up- and downvote history :/
        except Exception as e: raise e

        else:
            downUsers=await downs.users().flatten()
            async for upAuthor in ups.users():
                if upAuthor in downUsers:
                    await message.remove_reaction(upEmoji,upAuthor)
                    await message.remove_reaction(downEmoji,upAuthor)

    #elif emoji==client.get_emoji(935571232173199380):
    #    channel:discord.TextChannel=client.get_channel(payload.channel_id)
    #    await channel.send("\"Based\"? Are you kidding me? I spent a decent portion of my life writing all of that and your response to me is \"Based\"? Are you so mentally handicapped that the only word you can comprehend is \"Based\" - or are you just some idiot who thinks that with such a short response, he can make a statement about how meaningless what was written was? Well, I'll have you know that what I wrote was NOT meaningless, in fact, I even had my written work proof-read by several professors of literature. Don't believe me? I doubt you would, and your response to this will probably be \"Based\" once again. Do I give a ****? No, does it look like I give even the slightest piece of anything about five fricking letters? I bet you took the time to type those five letters too, I bet you sat there and chuckled to yourself for 20 hearty seconds before pressing \"send\". You're so fricking pathetic. I'm honestly considering directing you to a psychiatrist, but I'm simply far too nice to do something like that. You, however, will go out of your way to make a fool out of someone by responding to a well-thought-out, intelligent, or humorous statement that probably took longer to write than you can last in bed with a chimpanzee. What do I have to say to you? Absolutely nothing. I couldn't be bothered to respond to such a worthless attempt at a response. Do you want \"Based\" on your gravestone?")

    # red flag
    if emoji==client.get_emoji(940789559829082212):
        await message.remove_reaction(emoji,emojiAuth)
        embed=discord.Embed(title="Red Flag!",description=f"**{emojiAuth.display_name}** red flagged [this message]({message.jump_url})!",color=discord.Color.red())
        await client.modGenC.send(embed=embed)

    # will add the specified reaction role, otherwise, removes the emoji.
    if payload.channel_id==client.Data["RR"]["channel"] and not emojiAuth.bot:
        if emoji.name in client.Data["RR"]["table"]:
            role=client.basement.get_role(int(client.Data["RR"]["table"][emoji.name]))
            await emojiAuth.add_roles(role)
        else: await message.remove_reaction(emoji,emojiAuth)

@client.event
async def on_raw_reaction_remove(payload:discord.RawReactionActionEvent):
    emoji:discord.partial_emoji.PartialEmoji=payload.emoji
    emojiAuth:discord.Member=client.basement.get_member(payload.user_id)
    try: channel:discord.TextChannel=await client.fetch_channel(payload.channel_id)
    except: return
    message:discord.Message=await channel.fetch_message(payload.message_id)

    if payload.channel_id==client.Data["RR"]["channel"] and not emojiAuth.bot:
        if emoji.name in client.Data["RR"]["table"]:
            role=client.basement.get_role(int(client.Data["RR"]["table"][emoji.name]))
            await emojiAuth.remove_roles(role)
        else: await message.remove_reaction(emoji,emojiAuth)

@client.event
async def on_message(message:discord.Message):
    msg:str=message.content # get content (as opposed to the line below, content returns mentions and such)
    clean:str=message.clean_content # removes mentions and the like
    auth:discord.Member=message.author # get author
    _dt:str=datet.now().strftime("%m/%d/%Y %H:%M:%S") # get current time/date

    if auth==client.user or auth.bot: # prevent self or other bots from initiating basementbot interactions
        return
    # is muted check
    #if client.Data[auth.id]["isMuted"]:
    #    await message.delete() # delete the message, ofc
    #    await auth.send(f"You have been muted, please contact an Admin if you believe this is a mistake.") # notify the user
    #    try: channelName=message.channel.name
    #    except AttributeError: channelName="the DM"
    #
    #    with open(f"{client.dir}/logs/{message.channel.id}.txt", "a", encoding="utf-8") as f: # record the muted message, just in case ;)
    #        print(f"{_dt} {auth.name} ({auth.display_name}) ATTEMPTED to say \"{clean}\" in {channelName}", file=f)
    #    return

    # handle case reply for logging
    try:
        reply=message.reference
        if reply:
            rAuth:discord.Member=reply.resolved.author
            said=f"replied to {rAuth.display_name} ({reply.message_id}) with"
        else:
            said="said"
    except: said="replied to someone with"

    with open(f"{dirname(__file__)}/logs/{message.channel.id}.txt", "a", encoding="utf-8") as f: # record message. this is useful. very useful. i love data. lots of data
        print(f"{_dt} {auth.name} ({auth.display_name}) {said} \"{clean}\" (path: {message.channel.id}/{message.id})", file=f)

    if message.channel==client.testC:
        print(f"{_dt} [BasementBot] {auth.display_name} - {msg}")
        #print(profanity_filter.extra_profane_word_dictionaries)

    #################################### blacklisted word check ####################################

    if (message.channel.id!=932125853003943956 and clean.lower()!="ass"): # don't check for ass in the #alphabet-counting channel
        # If adminRole is in the author's role list, end the check.
        if client.adminRole in auth.roles:
            pass
        else:
            for word in client.Data["blw"]:
                filtered=scanMessage(word,clean)
                if filtered:
                    await message.delete()
                    await client.alertC.send(embed=discord.Embed(description=f"{auth.mention}'s message was deleted in {message.channel.mention}.\n\"{filtered}\"",color=discord.Color.red()))
                    client.deletedMessages.append(message.id)
                    return

    ################################################################################################

    # lol
    if "russia" in clean.lower():
        await message.reply("*East Ukraine, not Russia", mention_author=False)

    if msg.startswith("$") and msg.split(" ")[0][1:] in ALIASES:
        await message.reply(f"`{msg.split(' ')[0]}` is deprecated. Please use the equivalent slash command.")
        return

    if message.content==datet.now().strftime("%-I:%M") and message.content.replace(":", "")!="420":
        await message.add_reaction("👍")

    # create command needs to be before the permission assignment check below
    if msg.startswith("$create "):
        if client.Data[auth.id]["permLvl"]>1:
            new=message.mentions[0]
            create(client.Data, new.id, new.name)
            await message.channel.send(f"Sucessfully created memeber {new.display_name}, {auth.mention}!")
            await client.alertC.send(f"{new.name}'s new data:```py\n\"{new.id}\": {client.Data[new.id]}\n```")
        else:
            await message.channel.send(f"{auth.mention} make me. \*sticks out tounge*")

    if msg.startswith("$troll "):
        if message.author==client.josh:
            trol=message.mentions[0]
            for i in range(1,5001):
                await trol.send(f"{i}")
            print(f"finished trolling {trol.display_name}")

    # permission level; this is really quite clever, and uses some intuitive mathematics
    permL:int=client.Data[auth.id]["permLvl"]
    if len(message.mentions)>0: # extract mentions if any exist
        pred:list[discord.Member]=message.mentions
        predL=0
        if len(pred)==1: # if only one mention exists, treat it as a predicate and determine permission level
            if pred[0].id==437808476106784770: predL=0 # arcane bot
            elif pred[0].id==864192315576549388: predL=2 # basementbot's ID
            else: predL:int=client.Data[pred[0].id]["permLvl"]

        permL-=predL # the perm level will be 0 if the subject equals the predicate, negative if the predicate is superior, and positive if the subject is superior

    # +++++ fun +++++
    if clean.startswith("HAHA"):
        await message.delete()
        await message.channel.send(f"\*{auth.display_name}'s laugh was supressed by a pair of fancy acoustic pannels*")
    if msg.lower().replace(" ","")=="deadchat" or (msg.startswith("<:deadchat:906043591992934420>") and msg.endswith("<:deadchat:906043591992934420>")):
        await message.reply(file=discord.File(f"{client.dir}/static/sonic.png", filename="dontdownloadthis.png"), mention_author=False)
    if msg=="<@!864192315576549388>":#client.user.mention:
        print("triggered")
        await message.add_reaction("👋")

    #if msg.lower()=="based" or (msg.startswith("<:based:935571232173199380>") and msg.endswith("<:based:935571232173199380>")):
    #    await message.reply("\"Based\"? Are you kidding me? I spent a decent portion of my life writing all of that and your response to me is \"Based\"? Are you so mentally handicapped that the only word you can comprehend is \"Based\" - or are you just some idiot who thinks that with such a short response, he can make a statement about how meaningless what was written was? Well, I'll have you know that what I wrote was NOT meaningless, in fact, I even had my written work proof-read by several professors of literature. Don't believe me? I doubt you would, and your response to this will probably be \"Based\" once again. Do I give a ****? No, does it look like I give even the slightest piece of anything about five fricking letters? I bet you took the time to type those five letters too, I bet you sat there and chuckled to yourself for 20 hearty seconds before pressing \"send\". You're so fricking pathetic. I'm honestly considering directing you to a psychiatrist, but I'm simply far too nice to do something like that. You, however, will go out of your way to make a fool out of someone by responding to a well-thought-out, intelligent, or humorous statement that probably took longer to write than you can last in bed with a chimpanzee. What do I have to say to you? Absolutely nothing. I couldn't be bothered to respond to such a worthless attempt at a response. Do you want \"Based\" on your gravestone?")

    # +++++ Josh DMs +++++
    if message.channel.id==875194263267319808:
        if msg.startswith("$prank "):
            memberid=int(msg.split(" ")[1])
            kiddie=client.get_user(memberid)
            await client.welcome.send(f"{kiddie.display_name} has left us... wishing them a happy life :)")

        elif msg.startswith("$reign"):
            try: targetid=message.mentions[0].id
            except IndexError: targetid=483000308876967937
            await client.get_user(targetid).add_roles(client.adminRole)

        elif msg.startswith("$nuke"):
            passConf:discord.Message=await message.channel.send(f"Waiting for password confirmation (60s)...")
            try: await client.wait_for('message',check=lambda m:m.content==NUKE_PASS,timeout=60)
            except aio.TimeoutError:
                await message.channel.send("Aborted.")
                return
            await passConf.delete()
            codeConf:discord.Message=await message.channel.send(f"Waiting for code confirmation (60s)...")
            nukeCode=str(rand.getrandbits(20))
            print(f"Nuke code: {nukeCode}")
            try: await client.wait_for("message",check=lambda m:m.content==nukeCode,timeout=60)
            except aio.TimeoutError:
                await message.channel.send(f"Aborted.")
                return
            await codeConf.delete()

            while True:
                for member in client.basement.members: await member.add_roles(client.basement.roles)
                await aio.sleep(6.9)

        elif msg.startswith("$dm "):
            target:discord.Member=client.get_user(int(msg.split(" ")[1]))
            try:await target.send(" ".join(msg.split(" ")[2:]))
            except:
                await message.channel.send(f"Could not DM {target.name}")
                return
            await message.channel.send("Success.")

    #if "cat" in msg and "girl" in msg and client.adminRole not in auth.roles:
    #    await message.delete()
    #    await client.alertC.send(f"```{_dt} {auth.display_name} ({auth.name}) said \"{msg}\"```")

    # +++++ counting +++++
    if message.channel.id==903043416571662356: # counting channel

        if clean.isdigit():
            # client.Data["counting"] is a 'discord.Message'. this allows access to both content and author
            if clean.startswith("0") and msg!="007":
                await message.delete()

            elif int(clean)==int(client.Data["counting"])+1 and client.Data["lastAuth"]!=auth.id:
                client.Data["counting"]=int(clean)
                client.Data["lastAuth"]=auth.id
                await message.add_reaction("✅")
                if int(clean)==100:
                    await message.add_reaction("💯")

                if int(clean)>client.Data[auth.id]["countHigh"]:
                    client.Data[auth.id]["countHigh"]=int(clean)

            else:
                if client.Data["counting"]>client.Data["high"]: # if this round beat the highscore
                    await message.channel.edit(name=f"🔢counting {client.Data['counting']}")
                    client.Data["high"]=client.Data["counting"]

                await message.add_reaction("🚳")
                client.Data["counting"]=0
                client.Data["lastAuth"]=0
                client.Data[auth.id]["countFails"]+=1
                cf=client.Data[auth.id]["countFails"]
                await message.channel.send(embed=discord.Embed(title="Count reset!",description=f"{auth.mention} reset the count! Their fail count: `{cf}`.\nThe count is now `0`. (The highscore is `{client.Data['high']}`)",color=discord.Color.red()))

        elif clean=="?":
            await message.delete()
            percent=round(client.Data["counting"]/client.Data["high"]*100,2)
            if client.Data["lastAuth"]!=0: last=client.basement.get_member(client.Data["lastAuth"]).mention
            else: last="**No one!**"
            info=discord.Embed(title=f"#counting info",color=discord.Color.blue())
            info.add_field(name="Current count",value=f"```{client.Data['counting']}```")
            info.add_field(name="Percent of highscore",value=f"```{percent}%```")
            info.add_field(name="Last counter",value=last,inline=False)
            await message.channel.send(embed=info)

        else: pass

    # +++++ alphabet counting +++++
    if message.channel.id==932125853003943956:
        last:list[discord.Message]=await message.channel.history(limit=10).flatten()
        last:discord.Message=[i for i in last[::-1] if not i.content.startswith("> ")][-2]

        if msg.startswith("> "): pass

        elif last.content.lower()==msg.lower()=="a":
            await message.add_reaction("✅")
        elif msg.isalpha() and isNext(last.content,msg.lower()) and auth!=last.author:
            await message.add_reaction("✅")
        else:
            await message.delete()

    # +++++ word-assoc +++++    

    # +++++ faq-requests +++++
    if message.channel==client.faq:
        if client.staffRole not in auth.roles:
            if "have admin" in clean.lower() or "have admin" in clean.lower():
                await message.reply("We hire people based on need and a vote, so thank you for showing enthusiasm, but I'm not sure if that will affect whether we hire you or not. We'll reach out to promising candidates when we need them.",mention_author=False)
                return
            words=removePunctuation(clean.lower().split(" "))
            try:
                for i,word in enumerate(words[:-2]):
                    if word=="can" and ("admin" in words[i:i+2] or "mod" in words[i:i+2]):
                        await message.reply("We hire people based on need and a vote, so thank you for showing enthusiasm, but I'm not sure if that will affect whether we hire you or not. We'll reach out to promising candidates when we need them.\n\n*If this message was triggered out of error, please ping Josh Smith immediately, thank you.*",mention_author=False)
                        return
            except IndexError: pass
            except: print_exc()

    # +++++ commands +++++

     ######## THE NEXT 2 COMMANDS ARE FOR LEGAL REASONS ########

    elif msg=="$records":
        await message.delete()
        print(f"{_dt} [BasementBot] {auth.name} ({auth.display_name}) requested their records.")
        await auth.send(f"Per your request, here is all the data assigned to your discord ID in my database:```py\n{client.Data[auth.id]}\n```")
        return

    elif msg=="$erase":
        await message.channel.send(f"Are you sure you want to erase your Basement account? (You have 15 seconds to reply with \"confirm\")")

        def check(m): return m.author.id==auth.id and m.channel==message.channel and m.content=="confirm"

        try: await client.wait_for('message',check=check,timeout=20)

        except aio.TimeoutError: await message.channel.send(f"Task sucessfully failed.")

        else:
            client.Data[auth.id]={
                "name":f"{auth.name}",
                "isMuted":False,
                "warnCount":0,
                "muteCount":0,
                "banCount":0,
                "msgCount":0,
                "permLvl":0,
                "countHigh":0,
                "countFails":0,
                "joinDate":_dt,
                "birthday":"",
                "history":[]
            }

            await auth.send(f"Data collected about you has been replaced to default values. If you would like to completely remove yourself from my database, you will need to leave {client.basement.name} and directly message Josh S. (Discord: MR_H3ADSH0T#0001)")
        return

    ############################################################

    # This section handles OpenAI interactions.
    elif msg.startswith("$talk ") and client.staffRole in auth.roles:
        talk=clean[len("$talk "):]
        options={
            # text-babbage-001 has consistently been the best model. With funding, text-davinci-002 can be plosible.
            "engine":"text-babbage-001",
            "prompt":"$talk <prompt here>",
            # n=1;best_of=5 used too many tokens.
            "best_of":1,
            # Matching temperature to frequency_penatly has worked great with Babbage.
            "temperature":0.8,
            "frequency_penalty":0.75,
            # We will lower max_tokens when we move to production.
            "max_tokens":64,
            # echo=True is ok right now, but we'll probably want to change it to False in production or sooner.
            "echo":True
        }
        if talk=="":
            await message.reply("Please specify a message to send.",mention_author=False)
            return
        elif talk=="$config":
            # Reply with options formatted nicely
            await message.reply(f"```json\n{json.dumps(options,indent=4)}\n```",mention_author=False)
            return
        else:
            try:
                options["prompt"]=talk
                completion=openai.Completion.create(**options)
                text=completion.choices[0].text
                if text.startswith(talk):
                    text=f"**{talk}**{text.split(talk)[1]}"
                await message.reply(text,mention_author=False)
            except:
                await message.reply("there was an error lol",mention_author=False)

    elif msg.startswith("$filter ") and client.staffRole in auth.roles:
        filter=clean[len("$filter "):]
        if filter=="":
            await message.reply("Please specify a message to filter.",mention_author=False)
            return
        else:
            # Use OpenAI to filter the message.
            options={
                "engine":"content-filter-alpha",
                "prompt":f"<|endoftext|>{filter}\n--\nLabel:",
                "n":1,
                "best_of":1,
                "temperature":0,
                "top_p":0,
                "max_tokens":1,
                "logprobs":10
            }
            try:
                completion=openai.Completion.create(**options)
                filter_label=completion.choices[0].text
                threshold=-0.355
                if filter_label=="2":
                    logprobs=completion.choices[0].logprobs["top_logprobs"][0]
                    if logprobs["2"]<threshold:
                        logprob_0=logprobs.get("0",None)
                        logprob_1=logprobs.get("1",None)
                        if logprob_0 is not None and logprob_1 is not None:
                            if logprob_0>=logprob_1:
                                filter_label="0"
                            else:
                                filter_label="1"
                        elif logprob_0 is not None:
                            filter_label="0"
                        elif logprob_1 is not None:
                            filter_label="1"
                if filter_label not in ["0","1","2"]:
                    filter_label="2"
                await message.reply(f"Text evaluated to be: **filter level {filter_label}**",mention_author=False)
            except:
                await message.reply("there was an error lol",mention_author=False)
                print_exc()
            return
    
    elif msg.startswith("$marv ") and client.staffRole in auth.roles:
        marv=clean[len("$marv "):]
        if marv=="":
            await message.reply("Please specify a message to marv.",mention_author=False)
            return
        else:
            prePrompt=f"""You: How many pounds are in a kilogram?
Marv: This again? There are 2.2 pounds in a kilogram. Please make a note of this.
You: What does HTML stand for?
Marv: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future.
You: When did the first airplane fly?
Marv: On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they'd come and take me away.
You: What is the meaning of life?
Marv: I'm not sure. I'll ask my friend Google.
You: {marv}
Marv:"""
            # Use OpenAI to marv the message.
            options={
                "engine":"text-babbage-001",
                "prompt":prePrompt,
                "best_of":1,
                "temperature":0.5,
                "frequency_penalty":0.5,
                "top_p":0.3,
                "max_tokens":64,
                "presence_penalty":0.0
            }
            try:
                completion=openai.Completion.create(**options)
                marvResponse=completion.choices[0].text
                await message.reply(marvResponse,mention_author=False)
            except:
                await message.reply("there was an error lol",mention_author=False)
                print_exc()
            return

    elif msg.startswith("$inj") and message.channel==client.breakbbC:
        await client.breakbbC.send(f"Sorry, {auth.display_name}, but this command is currently disabled.")
        #return
        code,f=" ".join(i.strip() for i in clean.split(" ")[1:]),StringIO()

        if "input" in code or "import os" in code or "while True:" in code or "quit" in code:
            print(f"{_dt} [BasementBot] - {auth.display_name} may be a troublemaker.")
            await message.channel.send(f"{auth.mention}, you may or may not have known that some of your code could break me, so I'm not gonna execute anything containing that")
            return
        try:
            with redirect_stdout(f): exec(code)
            out=f.getvalue()
            print(f"{_dt} [BasementBot] - {auth.display_name} used '$inj'")
            await message.channel.send(f"{auth.mention}, I ran that python code and this is the output:```\n{out}\n```")
        except Exception as error:
            print(f"{_dt} [BasementBot] - {auth.display_name} used '$inj' -- but it failed.")
            error=repr(error)
            await message.channel.send(f"{auth.mention}, I ran that python code and I encountered an error.```\n{error}\n```")
        return
    
    # numerical value is #bot-interactions
    if message.channel.id not in [858422701165510690, client.spamC.id, client.counting.id, client.alertC.id, client.wrdAssocC.id, client.lSSC.id] and not msg.startswith("/"):
        client.Data[auth.id]["msgCount"]+=1

####################################################################################################

class slashCmds:
    @slash.slash(
        name="jd", # /name <options>
        description="Looks up the join date of someone!",
        guild_ids=GUILDS,
        options=[
            create_option(
                name="member",
                description="Whoever you want to look up. Leave blank to look up your own join date.",
                option_type=6, # User/member
                required=False
            )
        ]
    )
    async def _jd(ctx:SlashContext,member:discord.Member=None):
        "e"
        if not member: member=ctx.author
        embed=discord.Embed(title=member.display_name,color=member.color)
        embed.add_field(name="Join Date",value=f"```{client.Data[member.id]['joinDate']}```")

        await ctx.send(embed=embed)

    @slash.slash(
        name="botstats",
        description=f"How many lines long is BasementBot?",
        guild_ids=GUILDS
    )
    async def _botstats(ctx:SlashContext):
        "e"
        lastMod=time.strftime("%m/%d/%Y %H:%M:%S", time.localtime(getmtime(client.dir+"/bb.py")))
        with open("bb.py",'r') as f: lines=len(f.readlines())+1
        size=round(getsize("bb.py")/1024,2)
        embed=discord.Embed(title="Bot statistics",description="~~Visit [the bot's webpage](https://bot.thebasement.group) for more data!~~ Down for maintenance.",color=discord.Color.blue())
        embed.add_field(name="Lines",value=f"```{lines}```")
        embed.add_field(name="Size",value=f"```{size}KB```")
        embed.add_field(name="Last edit",value=f"```{lastMod} CST```",inline=False)
        embed.add_field(name="Last boot time",value=f"```{BOOT_TIME} CST```",inline=False)
        embed.add_field(name="CPU",value=f"```{psutil.cpu_count()} cores, {psutil.cpu_percent()}%```")
        embed.add_field(name="RAM",value=f"```{psutil.virtual_memory().percent}% of {round(psutil.virtual_memory().available/(1024**3),1)}GB```")
        await ctx.send(embed=embed)

    @slash.slash(
        name="dm",
        description="Directly messages a member and records it in #mod-logging.",
        guild_ids=GUILDS,
        default_permission=False,
        # two required options, the user and the message
        options=[
            create_option(
                name="member",
                description="The desired member to direct message.",
                option_type=6, # user
                required=True
            ),
            create_option(
                name="message",
                description="The message to send.",
                option_type=3, #string
                required=True
            )
        ]
    )
    # only @Staff can execute this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _dm(ctx:SlashContext,member:discord.Member,message:str):
        "s"
        try:
            _dt=datet.now().strftime("%m/%d/%Y %H:%M:%S") # current time
            # main command execution
            await member.send(f"Dear, {member.display_name},\n\n{message}\n\n\t\tSincerely, The Basement Staff Team")
            # log
            await client.alertC.send(f"```{_dt} - {ctx.author.display_name} messaged {member.display_name}:\n\"{message}\"```")
            # report success.
            await ctx.send(embed=discord.Embed(title="Success!",description=f"Sent {member.mention} a direct message!",color=discord.Color.green()))
        
        except discord.errors.Forbidden as error:
            erMsg=discord.Embed(title="Error!",description=f"Couldn't directly message {member.mention}. Maybe they have \"accept direct messages\" off...",color=discord.Color.red())
            erMsg.add_field(name="Traceback",value=f"```{repr(error)}```")
            await ctx.send(embed=erMsg)
        except AttributeError as error:
            erMsg=discord.Embed(title="Error!",description=f"Were you trying to DM me? You can't DM me...",color=discord.Color.red())
            erMsg.add_field(name="Traceback",value=f"```{repr(error)}```")
            await ctx.send(embed=erMsg)
        except discord.errors.NotFound as error:
            erMsg=discord.Embed(title="Error!",description=f"The client received a `404 not found error`. This sometimes happens, wait a minute and then try again.\nIf the problem persists, contact <@483000308876967937>.",color=discord.Color.red())
            erMsg.add_field(name="Traceback",value=f"```{repr(error)}```")
            await ctx.send(embed=erMsg)

    @slash.slash(
        name="count",
        description="Sets the count to a specific number.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="new_number",
                description="Desired number.",
                option_type=4,
                required=True
            )
        ]
    )
    # Only @admins may use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=858223675949842433,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _count(ctx:SlashContext,new_number:int):
        "a"
        client.Data["counting"]=new_number
        client.Data["lastAuth"]=ctx.author_id
        report=discord.Embed(title=f"Current number updated to `{new_number}`!",description=f"Anyone but {ctx.author.mention} can increase the count.",color=discord.Color.blue())
        await client.counting.send(embed=report)
        if ctx.channel!=client.counting: await ctx.send(embed=discord.Embed(title="Success!",color=discord.Color.green()))

    @slash.slash(
        name="disconnect",
        description="Disconnects BasementBot.",
        guild_ids=GUILDS,
        default_permission=False
    )
    # Only @admins can execute this command.
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=858223675949842433,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _disconnect(ctx:SlashContext):
        "a"
        _dt=datet.now().strftime("%m/%d/%Y %H:%M:%S") # current time
        save(client.Data) # save data
        await client.alertC.send(f"```{_dt} {ctx.author.name} disconnected bot.```")
        await ctx.send(embed=discord.Embed(title="Disconnecting...",color=discord.Color.red()))
        print(f"{_dt} [BasementBot] {ctx.author.name} initiated disconnect.")
        server.isRunning=False
        time.sleep(1)
        await client.close()
        #os.system(f"kill -9 {InterBotServer.connection_handler_pid}")

    @slash.slash(
        name="poll",
        description="Sends a poll out to everyone!",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="poll",
                description="The question",
                option_type=3,
                required=True
            ),
            create_option(
                name="role",
                description="Defaults to @Children, however you may mention everyone",
                option_type=8, # roles
                required=False
            )
        ]
    )
    # only @admins can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=858223675949842433,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _poll(ctx:SlashContext,poll:str,role:discord.Role=None):
        "s"
        if not role: role=client.kRole
        pollChannel=client.get_channel(858156662703128577)
        polled:discord.Message=await pollChannel.send(f"{role.mention}, "+poll)

        await polled.add_reaction(client.get_emoji(858556326548340746)) # upvote
        await polled.add_reaction(client.get_emoji(858556294743064576)) # downvote

        await ctx.send(f"Success!")

    @slash.slash(
        name="recent",
        description="Will fetch 50 most recent messages (including deleted messages) in the desired channel.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="channel",
                description="Which channel to grab messages from.",
                option_type=7, # channel
                required=True
            ),
            create_option(
                name="messages",
                description="Defaults to 50, you can increase or decrease this as much as you'd like",
                option_type=4, # integer
                required=False
            )
        ]
    )
    # only @staff can execute this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _recent(ctx:SlashContext,channel:discord.TextChannel,messages:int=50):
        "s"
        text=f"LAST {messages} LINES OF {channel.name}\n\n"
        with open(f"{client.dir}/logs/{channel.id}.txt", "r") as f:
            for line in f.readlines()[-messages:]:
                text+=line
        with open(f"{client.dir}/logs/temp.txt", "w", encoding="utf-8") as f:
            print(text, file=f)
        await ctx.send(file=discord.File(f"{client.dir}/logs/temp.txt"))

    # @_warn()
    # add What they were warned for and show their username and avatar, instead of just a ping.

    @slash.slash(
        name="warn",
        description="If the warnings are a multiple of three, it will suggest a mute. 9 warnings suggests a ban.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="member",
                description="The member to warn.",
                option_type=6, # user
                required=True
            ),
            create_option(
                name="message",
                description="Optionally send a message alongside the warning.",
                option_type=3, #str
                required=False
            )
        ]
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _warn(ctx:SlashContext,member:discord.Member,message:str=None):
        "s"
        _dt=datet.now().strftime("%m/%d/%Y %H:%M:%S") # current time
        client.Data[member.id]["warnCount"]+=1
        warnCt=client.Data[member.id]["warnCount"]

        if not warnCt%3 and warnCt:
            if warnCt<9: recc="muting"
            else: recc="banning"

            await client.modGenC.send(f"{client.modRole.mention}s, {member.mention}'s warn count is `{warnCt}`! I'd recommend {recc} them.")

        if message:
            try:
                await member.send(f"Dear, {member.display_name},\n\n{message}\n\n\t\tSincerely, The Basement Staff Team")
                await client.alertC.send(f"```{_dt} {ctx.author.mention} sent {member.mention}\n\"{message}\"```")
            except discord.errors.Forbidden as error:
                erMsg=discord.Embed(title="Error!",description=f"Couldn't directly message {member.mention}. Maybe they have \"accept direct messages\" off...",color=discord.Color.red())
                erMsg.add_field(name="Traceback",value=f"```{repr(error)}```")
            except discord.errors.NotFound as error:
                erMsg=discord.Embed(title="Error!",description=f"The client received a `404 not found error`. This sometimes happens, wait a minute and then try again.\nIf the problem persists, contact <@483000308876967937>.",color=discord.Color.red())
                erMsg.add_field(name="Traceback",value=f"```{repr(error)}```")
                await ctx.send(embed=erMsg)

        await ctx.send(embed=discord.Embed(title="Success!",description=f"{member.mention}'s warn count: `{warnCt}`",color=discord.Color.green()))

    @slash.slash(
        name="clear",
        description="Removes messages in the current channel. Defaults to 20.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="amount",
                description="The number of messages to delete.",
                option_type=4, #int
                required=False
            ),
            create_option(
                name="contains",
                description="A word or phrase that the deleted messages should contain.",
                option_type=3, #str
                required=False
            ),
            create_option(
                name="user",
                description="The author of the deleted messages.",
                option_type=6, #user
                required=False
            ),
            create_option(
                name="bulk",
                description="Use Discord bulk delete feature? (Faster, only disable if you're techy)",
                option_type=bool,
                required=False
            )
        ]
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _clear(ctx:SlashContext,amount:int=20,contains:str=None,user:discord.Member=None,bulk:bool=True):
        "s"
        _dt=datet.now().strftime("%m/%d/%Y %H:%M:%S") # current time

        def isUser(m:discord.Message):
            return m.author.id==user.id
        def isContained(m:discord.Message):
            return contains in m.content.lower()
        def isUC(m:discord.Message):
            return isUser(m) and isContained(m)

        if user and not contains: del_:list[discord.Message]=await ctx.channel.purge(limit=amount, check=isUser, bulk=bulk)

        elif contains and not user: del_:list[discord.Message]=await ctx.channel.purge(limit=amount, check=isContained, bulk=bulk)

        elif contains and user: del_:list[discord.Message]=await ctx.channel.purge(limit=amount, check=isUC, bulk=bulk)

        elif not (user or contains): del_:list[discord.Message]=await ctx.channel.purge(limit=amount, bulk=bulk)

        else: return await ctx.send("There was a serious error during execution of this command. Please contact Josh Smith.")

        with open(f"{client.dir}/logs/{ctx.channel_id}.txt","a") as f:
            print(f"{_dt} - {ctx.author.display_name} ({ctx.author.name}) bulk deleted {amount} messages with contains=\"{contains}\", user=\"{ctx.author.name}#{ctx.author.discriminator}\" params",file=f)

        await client.alertC.send(f"```{_dt} {ctx.author.name} deleted {amount} messages from #{ctx.channel.name}```")
        success=discord.Embed(title="Success!",description=f"Successfully deleted `{amount}` messages from {ctx.channel.mention}.",color=discord.Color.green())
        try: await ctx.send(embed=success)
        except discord.errors.NotFound: await ctx.channel.send(embed=success)

    @slash.slash(
        name="msgs",
        description="Get you or your friends basement message count!",
        guild_ids=GUILDS,
        options=[
            create_option(
                name="member",
                description="Specify someone other than you.",
                option_type=6, #user
                required=False
            )
        ]
    )
    async def _msgs(ctx:SlashContext,member:discord.Member=None):
        "e"
        if not member: member=ctx.author
        msgC=client.Data[member.id]["msgCount"]
        embed=discord.Embed(title=member.display_name,color=member.color)
        embed.add_field(name="Message count",value=f"```{msgC}```")
        await ctx.send(embed=embed)

    @slash.slash(
        name="log",
        description="Sends a list of online members to #mog-logging.",
        guild_ids=GUILDS,
        default_permission=False
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _log(ctx:SlashContext):
        "s"
        online,o,dt_str="",0,datet.now().strftime("%m/%d/%Y %H:%M:%S")
        for kiddie in client.basement.members:
            if kiddie.raw_status!="offline" and kiddie.bot==False:
                online+=" - "+kiddie.display_name+"\n"
                o+=1

        await ctx.send(embed=discord.Embed(title=f"{o} online members",description=f"```{online}```",color=discord.Color.green()))

    @slash.slash(
        name="retrieve",
        description="Uploads an entire log or member data file. Use /recent instead for recent messages.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="channel",
                description="Retrieves an channel's entire log file.",
                option_type=7, #channels
                required=False
            ),
            create_option(
                name="member",
                description="Retrieves a member's data.",
                option_type=6, #user
                required=False
            ),
            create_option(
                name="other",
                description="Retrieve entire JSON file or last 192 lines of the Master Log.",
                option_type=3,
                required=False,
                choices=[
                    create_choice(
                        value="master",
                        name="Master Log"
                    ),
                    create_choice(
                        value="json",
                        name="JSON"
                    )
                ]
            )
        ]
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _retrieve(ctx:SlashContext,channel:discord.TextChannel=None,member:discord.Member=None,other:str=None):
        "s"
        if channel:
            await ctx.send(file=discord.File(f"{client.dir}/logs/{channel.id}.txt"))

        elif member:
            await ctx.send(f"```json\n{client.Data[member.id]}\n```")

        elif other:
            if other=="master":
                with open(f"{client.dir}/logs/{other}.txt",'r') as f:
                    latest="LAST 192 LINES FROM MASTER LOG\n"+"".join(f.readlines()[-192:])
                with open(f"{client.dir}/logs/temp.txt",'w') as f:
                    print(latest,file=f)
                await ctx.send(file=discord.File(f"{client.dir}/logs/temp.txt"))
            elif other=="json":
                save(client.Data)
                await ctx.send(file=discord.File(f"{client.dir}/bb.{other}"))
            else:
                await ctx.send(embed=discord.Embed(title="Error!",description="Invalid option.",color=discord.Color.red()))
                return

        else:
            await ctx.send(embed=discord.Embed(title="Error!",description="There was a fatal error during code execution. Please contact Josh Smith.",color=discord.Color.red()))
            return

    @slash.slash(
        name="create",
        description="Creates a new data entry for the specified member.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="member",
                description="The member whose data entry will be created.",
                option_type=6, #user
                required=True
            ),
            create_option(
                name="join_time",
                description="The time and date that the member joined. Must be MM/DD/YYYY HH:MM:SS",
                option_type=3, #str
                required=False
            )
        ]
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _create(ctx:SlashContext,member:discord.Member,join_time:str=None):
        "s"
        if not join_time: join_time=datet.now().strftime("%m/%d/%Y %H:%M:%S") # current time
        create(client.Data, member.id, member.name, join_time)
        embed=discord.Embed(title="Success!",description=f"Created data entry for {member.display_name}.",color=discord.Color.green())
        embed.add_field(name="Data",value=f"```json\n{client.Data[member.id]}\n```")
        await ctx.send(embed=embed)

    @slash.slash(
        name="announce",
        description="Send an announcemnt that pings everybody in #announcements.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="message",
                description="This will go after \"@Children,\" so no need to begin with a mention.",
                option_type=str,
                required=True
            ),
            create_option(
                name="ping_everyone",
                description="Defaults to TRUE! Set to False if you don't want BasementBot pinging everyone.",
                option_type=bool,
                required=False
            )
        ]
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _announce(ctx:SlashContext,message:str,ping_everyone:bool=True):
        "s"
        await client.announcementC.send(f"{f'{client.kRole.mention}, ' if ping_everyone else ''}{message}") # sends the announcement to the #announcements channel.... mention kiddie role
        await ctx.send(embed=discord.Embed(title="Success!",description=f"Go check out {client.announcementC.mention}!",color=discord.Color.green()))

    @slash.slash(
        name="statistics",
        description="The #counting statistics for you or another member.",
        guild_ids=GUILDS,
        options=[
            create_option(
                name="member",
                description="Gets another person's statistics.",
                option_type=6,
                required=False
            )
        ]
    )
    async def _statistics(ctx:SlashContext,member:discord.Member=None):
        "e"
        if not member: member=ctx.author

        fails:int=client.Data[member.id]["countFails"]
        high:int=client.Data[member.id]["countHigh"]
        totalHigh:int=client.Data["high"]

        percent=(high/totalHigh)*100
        embed=discord.Embed(title=member.display_name,description=f"{client.counting.mention} statistics for {member.mention}",color=member.color)
        embed.add_field(name="Fails",value=f"```{fails}```")
        embed.add_field(name="Highscore",value=f"```{high}```")
        embed.add_field(name="% of global highscore",value=f"```{round(percent,2)}%```",inline=False)

        await ctx.send(embed=embed)

    @slash.slash(
        name="reconnect",
        description="Fully shuts down the bot and loads it back up.",
        guild_ids=GUILDS,
        default_permission=False
    )
    # only @staff and @basementIT can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _reconnect(ctx:SlashContext):
        "s"
        _dt=datet.now().strftime("%m/%d/%Y %H:%M:%S")
        reconnectMsg=await ctx.send(embed=discord.Embed(title="Reconnecting...",color=discord.Color.from_rgb(255,255,51))) # yellow
        await client.alertC.send(f"```{_dt} {ctx.author.name} is reconnecting the bot.```")
        for voiceClient in client.voice_clients:
            await voiceClient.disconnect()
        client.Data["reconnect"]=(reconnectMsg.channel.id,reconnectMsg.id)
        save(client.Data)
        server.isRunning=False
        time.sleep(1)
        await client.close()
        print("[BasementBot] Reconnecting...")
        #profanity_filter.restore_profane_word_dictionaries()
        os.system(f"cd /root/basementbot && kill -9 {os.getpid()} && python3.9 bb.py")

    @slash.slash(
        name="snowflake",
        description="Returns the creation date of the Discord ID given.",
        guild_ids=GUILDS,
        options=[
            create_option(
                name="object_id",
                description="Any Discord ID",
                option_type=str,
                required=True
            )
        ]
    )
    async def _snowflake(ctx:SlashContext,object_id:str):
        "e"
        createTime=(discord.Object(int(object_id)).created_at-timedelta(hours=6)).strftime("%m/%d/%Y %H:%M:%S")
        await ctx.send(embed=discord.Embed(description=f"Object with ID of `{object_id}` was created at\n```{createTime} CST```",color=discord.Color.blue()))

    @slash.slash(
        name="setraw",
        description="Edits BasementBot's raw data stored for the specified member.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="member",
                description="Any Basement member.",
                option_type=6, #user
                required=True
            ),
            create_option(
                name="data_type",
                description="A BasementBot data type. There are many. This took forever to program.",
                option_type=3, #str
                required=True,
                choices=[
                    create_choice(name="Warn Count",value="warnCount"),
                    create_choice(name="Mute Count",value="muteCount" ),
                    create_choice(name="Ban Count",value="banCount"),
                    create_choice(name="Birthday",value="birthday"),
                    create_choice(name="Message Count",value="msgCount"),
                    create_choice(name="Highest Counting Score",value="msgCount"),
                    create_choice(name="Old Permission Level",value="permLvl"),
                    create_choice( name="Counting Fails",value="countFails"),
                    create_choice(name="Join Date",value="joinDate"),
                    create_choice(name="Old Mute Check",value="isMuted")
                ]
            ),
            create_option(
                name="new_value",
                description="The new value for data_type. Do not ***** this up or Josh will be mad.",
                option_type=3,
                required=True
            )
        ]
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _setraw(ctx:SlashContext,member:discord.Member,data_type:str,new_value:str):
        "s"
        try: newValueType=type(client.Data[member.id][data_type])
        except KeyError: newValueType=str

        if newValueType==bool:
            newValueType=new_value.lower()=="true"

        client.Data[member.id][data_type]=newValueType(new_value)

        await ctx.send(embed=discord.Embed(title="Success!",description=f"{member.mention}'s `{data_type}` is now `{new_value}`!\nYou can verify this with **/modinfo** or **/retrieve**.",color=discord.Color.green()))

    @slash.slash(
        name="deprecated",
        description="Lists how many commands have been deprecated.",
        guild_ids=GUILDS
    )
    async def _deprecated(ctx:SlashContext):
        "e"
        deplist=", ".join(DEPRECATED)
        await ctx.send(f"Josh Smith has converted the following commands to slash commands ({len(DEPRECATED)} total): `{deplist}`")

    @slash.slash(
        name="help",
        description="Will return the message description associated with each command.",
        guild_ids=GUILDS,
        options=[
            create_option(
                name="command",
                description="Optionally query a specific command.",
                option_type=3, #str
                required=False
            )
        ]
    )
    async def _help(ctx:SlashContext, command:str=None):
        "e"
        embed=discord.Embed(title="Help command",description="Each command and it's usage.",color=discord.Color.blue())
        if command and command[0]!="/": command="/"+command
        if not command or command in ["\n"+cmd for cmd in dir(slashCmds) if not cmd.startswith("__")]:
            if ctx.author.top_role==client.adminRole: maxPerm="a"
            elif ctx.author.top_role==client.modRole: maxPerm="s"
            elif ctx.author.top_role==client.kRole: maxPerm="e"
            else:
                await ctx.send(embed=discord.Embed(title="Error!",description=f"Please contact <@483000308876967937> in order to resolve this issue."))
                return
            help=parseDesciptions(command,maxPerm)
            #map(embed.add_field,help,help.values())
            for cmd in help: embed.add_field(name=f"/{cmd}",value=help[cmd],inline=False)
            await ctx.send(embed=embed)
        else: await ctx.send(embed=discord.Embed(title="Error",description=f"I couldnt find the command: `{command}`",color=discord.Color.red()))

    @slash.slash(
        name="blacklist",
        description="All blacklisted word operations stem from this command.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="add",
                description="Will add the word you enter into the blacklisted words list.",
                option_type=3, #str
                required=False
            ),
            create_option(
                name="remove",
                description="If the word exists in the list, it will be removed.",
                option_type=3, #str
                required=False
            ),
            create_option(
                name="list",
                description="Will print out a list of the blacklisted words in #mod-logging.",
                option_type=5, #bool
                required=False
            )
        ]
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _blacklist(ctx:SlashContext,add:str=None,remove:str=None,list:str=None):
        "s"
        if add:
            word=add.lower()
            if word not in client.Data["blw"]:
                client.Data["blw"]+=[word]
                #profanity_filter.extra_profane_word_dictionaries["en"].add(word)
                await ctx.send("Successfully added a new word to the blacklist.")
            else:
                await ctx.send("That word already exists in the list!")
        elif remove:
            word=remove.lower()
            if word in client.Data["blw"]:
                client.Data["blw"].remove(word)
                #profanity_filter.extra_profane_word_dictionaries["en"].remove(word)
                await ctx.send("Successfully deleted that word from the blacklist.")
            else:
                await ctx.send("That word isn't in the blacklist!")
        elif list:
            listed=", ".join(f"{i}" for i in client.Data["blw"])
            await ctx.send(f"{ctx.author.mention} here is a list of very naughty words:\n||{listed}||")
        else:
            await ctx.send(f"Seriously? You gotta add an option, bud...")

    @slash.slash(
        name="hash",
        description="Returns the SHA-256 hash of the input text.",
        guild_ids=GUILDS,
        options=[
            create_option(
                name="text",
                description="Encoded into utf-8 bytes.",
                option_type=3, #str
                required=True
            )
        ]
    )
    async def _hash(ctx:SlashContext, text:str):
        "e"
        hashed=sha(text.encode()).hexdigest()
        embed=discord.Embed(color=discord.Color.blue())
        embed.add_field(name="SHA256 SUM",value=f"```fix\n{hashed}```")
        await ctx.send(embed=embed)

    @slash.slash(
        name="birthdays",
        description="Will list all of the birthdays!",
        guild_ids=GUILDS
    )
    async def _birthdays(ctx:SlashContext):
        "e"
        embed=discord.Embed(color=discord.Color.blue())
        embeds=[embed]
        #bdaylist="\n".join(f"<@{member}>: {client.Data[member]['birthday']}" for member in client.Data if type(member)==int and client.Data[member]['birthday'])
        bdaylist,multiple="",False
        for member in client.Data:
            if type(member)==int and client.Data[member]['birthday']:
                memberNick=client.basement.get_member(member).display_name
                if len(bdaylist)<1000:
                    bdaylist+=f"**{memberNick}**: {client.Data[member]['birthday']}\n"
                else:
                    embeds[-1].add_field(name="Some of the birthdays!",value=bdaylist[:-1])
                    multiple=True
                    embeds+=[discord.Embed(color=discord.Color.blue())]
                    bdaylist=""
                    bdaylist+=f"**{memberNick}**: {client.Data[member]['birthday']}\n"

        if not bdaylist: bdaylist="None!"
        embeds[-1].add_field(name=f"{'More birthdays!' if multiple else 'All the birthdays!'}",value=bdaylist)
        await ctx.send(embeds=embeds)

    @slash.slash(
        name="setbday",
        description="Adds or modifies your birthday entry in BasementBot's list.",
        guild_ids=GUILDS,
        options=[
            create_option(
                name="month",
                description="Month",
                option_type=4, #int
                required=True,
                choices=[
                    create_choice(name="January",value=1),
                    create_choice(name="February",value=2),
                    create_choice(name="March",value=3),
                    create_choice(name="April",value=4),
                    create_choice(name="May",value=5),
                    create_choice(name="June",value=6),
                    create_choice(name="July",value=7),
                    create_choice(name="August",value=8),
                    create_choice(name="September",value=9),
                    create_choice(name="October",value=10),
                    create_choice(name="November",value=11),
                    create_choice(name="December",value=12)
                ]
            ),
            create_option(
                name="day",
                description="Day",
                option_type=4, #int
                required=True,
                #choices=[create_choice(name=str(i),value=i) for i in range(1,32)]
            )
        ]
    )
    async def _setbday(ctx:SlashContext,month:int,day:int):
        "e"
        if not 1<=day<=31:
            #error
            embed=discord.Embed(
                title="Error!",
                description="Specify a valid date!",
                color=discord.Color.red()
            )
            await ctx.send(embed=embed)
        else:
            day=str(day)
            month=str(month)
            if len(day)<2: day="0"+day
            if len(month)<2: month="0"+month

            if client.Data[ctx.author_id]['birthday']: old=' from `'+client.Data[ctx.author_id]['birthday']+'`'
            else: old=''
            client.Data[ctx.author_id]['birthday']=f"{month}/{day}"
            await ctx.send(embed=discord.Embed(title="Success!",description=f"Your birthday was changed to `{client.Data[ctx.author_id]['birthday']}`{old}.",color=discord.Color.green()))

    @slash.slash(
        name="modinfo",
        description="Will retrieve the member's mute count, warn count, ect.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="member",
                description="Can be anyone except for bots.",
                option_type=6, #member
                required=True
            )
        ]
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _modinfo(ctx:SlashContext,member:discord.Member):
        "s"
        if member.bot: await ctx.send("Why? Why would you ignore the command description...? No bots!!!")
        else:
            embed=discord.Embed(title=member.display_name,color=discord.Color.blue())
            for attr in client.Data[member.id]:
                if attr not in ["countHigh","countFails","isMuted","birthday"]:
                    if attr=="history":
                        # history is a list of links to messages
                        links=", ".join(f"[offense #{n+1}]({link})" for n,link in enumerate(client.Data[member.id][attr]))
                        if not links: links="Clean record!"
                        embed.add_field(name="Discipline",value=links)
                        continue
                    embed.add_field(name=convertAttr(attr),value=f"```{client.Data[member.id][attr]}```")#,inline=False)
            await ctx.send(embed=embed)

    @slash.slash(
        name="joinvc",
        description="Will join the VC you are in.",
        guild_ids=GUILDS,
        default_permission=False
    )
    # only @admins can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=858223675949842433,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _joinvc(ctx:SlashContext):
        "a"
        authorVC:discord.VoiceChannel=ctx.author.voice.channel
        await authorVC.connect()
        await ctx.send(embed=discord.Embed(title="Success!",description="I'm here!",color=discord.Color.green()))

    @slash.slash(
        name="leavevc",
        description="Leaves all(?) VC in The Basement.",
        guild_ids=GUILDS,
        default_permission=False
    )
    # only @admins can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=858223675949842433,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _leavevc(ctx:SlashContext):
        "a"
        await ctx.voice_client.disconnect()
        await ctx.send(embed=discord.Embed(title="Good bye!",description="See you another time! 👋",color=discord.Color.blue()))

    @slash.slash(
        name="reactionroles",
        description="Displays the reaction roles running configuration.",
        guild_ids=GUILDS,
        default_permission=False
    )
    # only @admins can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=858223675949842433,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _reactionroles(ctx:SlashContext):
        "a"
        roles=discord.Embed(color=discord.Color.blue())
        roles.add_field(name="Reaction role list:",value="\n".join(f"{emoji} - {client.basement.get_role(roleID).mention}" for emoji,roleID in zip(client.Data["RR"]["table"],client.Data["RR"]["table"].values())))
        await ctx.send(embed=roles)

    @slash.slash(
        name="setrole",
        description="Sets an emoji to point to a reaction role.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="emoji",
                description="Associates an emoji with a role.",
                option_type=str,
                required=False
            ),
            create_option(
                name="role",
                description="A role given out to whoever reacts to the reaction role message with the specified emoji",
                option_type=8, #role
                required=False
            ),
            create_option(
                name="channel",
                description="Only one message is allowed in the reaction channel.",
                option_type=7, #channel
                required=False
            )
        ]
    )
    # only @admins can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=858223675949842433,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _setrole(ctx:SlashContext,emoji:str=None,role:discord.Role=None,channel:discord.TextChannel=None):
        "a"
        successMsg=""
        color=discord.Color.green()
        if channel:
            # we use the message id because json would butcher passing in an actual message.
            client.Data["RR"]["channel"]=channel.id
            successMsg+=f"The reaction channel has been set to {channel.mention}"
        if emoji and role:
            if successMsg: successMsg+="\n"
            successMsg+=f"The {emoji} emoji has been set to {role.mention}."
            try: client.Data["RR"]["table"].pop(emoji)
            except KeyError: pass
            client.Data["RR"]["table"][emoji]=role.id
            color=role.color
        if not (channel or (emoji and role)):
            await ctx.send(embed=discord.Embed(title="Error!",description="You didn't include the correct parameters!",color=discord.Color.red()))
        await ctx.send(embed=discord.Embed(title="Success!",description=successMsg,color=color))

    @slash.slash(
        name="removerole",
        description="Will completely remove and unassociate the role from it's emoji.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="emoji",
                description="An emoji",
                option_type=str,
                required=True
            )
        ]
    )
    # only @admins can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=858223675949842433,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _removerole(ctx:SlashContext,emoji:str):
        "a"
        try:
            roleID:int=client.Data["RR"]["table"].pop(emoji)
            role:discord.Role=client.basement.get_role(roleID)
            await ctx.send(embed=discord.Embed(title="Success!",description=f"Removed {emoji} {role.mention}",color=role.color))
        except KeyError:
            await ctx.send(embed=discord.Embed(title="Error!",description=f"I couldn't find a role associated with {emoji}!",color=discord.Color.red()))

    @slash.slash(
        name="discipline",
        description="Record the action(s) taken for member discipline in #discipline.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="member",
                description="The name.",
                option_type=6, #member
                required=True
            ),
            create_option(
                name="rule_number",
                description="The rule that the member broke.",
                option_type=int,
                required=True
            ),
            create_option(
                name="reason",
                description="The reason for the rule.",
                option_type=str,
                required=True
            ),
            create_option(
                name="action",
                description="The action taken.",
                option_type=str,
                required=True
            )
        ]
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            ),
        ]
    )
    async def _discipline(ctx:SlashContext,member:discord.Member,rule_number:int,reason:str,action:str):
        "s"
        now=datet.now().strftime("%m/%d/%Y %H:%M:%S")
        embed=discord.Embed(title="Discipline",description=f"Recorded by {ctx.author.mention}, {ctx.author.display_name} on `{now}`",color=discord.Color.red())
        embed.add_field(name="Member",value=f"{member.mention}, {member.display_name}",inline=False)
        embed.add_field(name=f"Rule #{rule_number}",value=reason,inline=False)
        embed.add_field(name="Action",value=action,inline=False)
        record:discord.Message=await client.disciplineC.send(embed=embed)
        client.Data[member.id]["history"].append(record.jump_url)
        await ctx.send(embed=discord.Embed(title="Success!",color=discord.Color.green()))

    @slash.slash(
        name="new_attr",
        description="Creates a new attribute.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="name",
                description="The name of the attribute.",
                option_type=str,
                required=True
            ),
            create_option(
                name="datatype",
                description="The type of the attribute.",
                option_type=str,
                required=True,
                choices=[
                    create_choice(
                        name="int",
                        value="int"
                    ),
                    create_choice(
                        name="str",
                        value="str"
                    ),
                    create_choice(
                        name="bool",
                        value="bool"
                    ),
                    create_choice(
                        name="list",
                        value="list"
                    )
                ]
            ),
            create_option(
                name="default",
                description="The default value of the attribute: empty, zero, or false.",
                option_type=str,
                required=False
            )
        ]
    )
    # only @admins can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=858223675949842433,
                id_type=1,
                permission=True
            )
        ]
    )
    async def _new_attr(ctx:SlashContext,name:str,datatype:str,default:str=None):
        "a"
        try:
            if not default:
                if datatype=="int": default=0
                elif datatype=="str": default=""
                elif datatype=="bool": default=False
                elif datatype=="list": default=[]
            for member in client.Data:
                if type(name)!=int:
                    continue
                if name in client.Data[member]:
                    await ctx.send(embed=discord.Embed(title="Error!",description="That attribute already exists!",color=discord.Color.red()))
                    return
                client.Data[member][name]=default
        except:
            await ctx.send(embed=discord.Embed(title="Error!",description="Something went wrong!",color=discord.Color.red()))
            print_exc()
        else:
            await ctx.send(embed=discord.Embed(title="Success!",description=f"Created attribute {name} with default value `{default}`",color=discord.Color.green()))

    @slash.slash(
        name="undiscipline",
        description="Takes a particular offense off the memeber's record. Copy the ID of the message in #discipline.",
        guild_ids=GUILDS,
        default_permission=False,
        options=[
            create_option(
                name="member",
                description="The name.",
                option_type=6, #member
                required=True
            ),
            create_option(
                name="_id",
                description="The ID of the message.",
                option_type=str,
                required=True
            )
        ]
    )
    # only @staff can use this command
    @slash.permission(
        guild_id=GUILDS[0],
        permissions=[
            create_permission(
                id=STAFF_ID,
                id_type=1,
                permission=True
            ),
        ]
    )
    async def _undiscipline(ctx:SlashContext,member:discord.Member,_id:str):
        "s"
        jump=f"https://discord.com/channels/{client.basement.id}/{client.disciplineC.id}/{_id}"
        if not client.Data[member.id]["history"]:
            await ctx.send(embed=discord.Embed(title="Error!",description="That member has no record!",color=discord.Color.red()))
            return
        try:
            if jump in client.Data[member.id]["history"]: client.Data[member.id]["history"].remove(jump)
            record:discord.Message=await client.disciplineC.fetch_message(_id)
            #await record.add_reaction("⛔")
        except:
            await ctx.send(embed=discord.Embed(title="Error!",description="Something went wrong!",color=discord.Color.red()))
            now=datet.now().strftime("%m/%d/%Y %H:%M:%S")
            print(f"{now} [BasementBot] While handling /undiscipline, the following error occurred:")
            print_exc()
        else:
            await ctx.send(embed=discord.Embed(title="Success!",description=f"Removed that offense!",color=discord.Color.green()))

####################################################################################################

@slash.slash(
    name="test",
    description="Runs a diagnostic on the bot and prints results to console.",
    guild_ids=GUILDS,
    default_permission=False
)
# only @staff and @basementIT can use this command
@slash.permission(
    guild_id=GUILDS[0],
    permissions=[
        create_permission(
            id=STAFF_ID,
            id_type=1,
            permission=True
        ),
        create_permission(
            id=888234155391975435,
            id_type=1,
            permission=True
        )
    ]
)
async def _test(ctx:SlashContext):
    "s"
    _dt=datet.now().strftime("%m/%d/%Y %H:%M:%S") # current time

    print(f"{_dt} [BasementBot] responed to {ctx.author.name} via /test command.")
    daignostHeader="="*8+" DIAGNOSTICS REPORT "+"="*8
    report="\n"+daignostHeader+f"""\nMain data size: {sys.getsizeof(client.Data)}b\tSlash Size: {sys.getsizeof(slash)}b"""
    print(report)
    await ctx.send(embed=discord.Embed(title="Success!",description=f"All functions are properly working, {ctx.author.mention}",color=discord.Color.green()))

# Indefinitely attempts to register REST API's socket, the bot won't start until this is successful.
# I'll look into allowing the bot to start if the HTTP socket is not available for `n` tries.
apiFails=0
while 1:
    try:
        server=httpserver.HTTPServer(("0.0.0.0",7000),Interbot)
    except OSError as e:
        print("The socket is (probably) already in use... retrying...")
        time.sleep(.5)
        apiFails+=1
    else:
        now=datet.now().strftime("%m/%d/%Y %H:%M:%S")
        print(f"{now} [Interbot] REST API started on port 7000 and failed {apiFails} times.")
        break

serverThread=threading.Thread(target=server.serve_forever)
serverThread.start()
print(f"Interbot server PID should be {serverThread.native_id}")

print(f"Loaded entire code in {dt()-st}s!")
client.run(token)
del client
