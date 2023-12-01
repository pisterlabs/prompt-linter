# -*- coding: utf-8 -*-\
import subprocess
from os.path import dirname
from os.path import realpath
import codecs
import json
import requests
import time
from os.path import exists
from urllib.parse import quote
dirpath = dirname(realpath(__file__)) + "/"
with codecs.open(dirpath + "atrikey.json", encoding='utf-8', mode='r') as r:
    # aaa=r.read().encode().decode('utf-8-sig') win7 workaround
    key = json.loads(r.read())
    name = key["name"]
    print("主人我的云控制链接是"+key["songctladdr"])
if not key["devmode"]:
    subprocess.Popen(["node",dirpath+"NeteaseCloudMusicApi/app.js"])
    cloudmusicapiurl = 'http://127.0.0.1:' + key["NeteaseCloudMusicApiPort"]
    print("主人，亚托莉已经帮你启动了网易云音乐API喵~")
else:
    print("主人，开发者模式以启用")
    cloudmusicapiurl = 'http://192.168.50.58:3000'

if not exists(dirpath + "cookie.txt"):
    time.sleep(10)
    loginres = requests.get(
        cloudmusicapiurl + "/login/cellphone?phone=" + key["NeteaseCloudMusicUsername"] + "&password=" + key[
            "NeteaseCloudMusicPassword"])
    cookie=quote(loginres.json()['cookie'])
    with codecs.open(dirpath + "cookie.txt", encoding='utf-8', mode='w') as f:
        f.write(cookie)
    print("主人，我已经登录网易云了并且保存了cookie喵~，cookie是" + quote(
        loginres.json()['cookie']))
else:
    with codecs.open("cookie.txt", encoding='utf-8', mode='r') as r:
        cookie = r.read()
from os import system as cmd
from prettytable import PrettyTable
from prettytable import PLAIN_COLUMNS
from flask import Flask
from flask import request
from flask import send_file
from flask import send_from_directory
import ast
import asyncio
import os
import platform
import random
import traceback
import aiofiles
from datetime import date
from functools import partial
import discord
import yt_dlp
from discord.ext import commands
from discord.ext import tasks
from gtts import gTTS
import openai
import threading
import json
from mutagen.mp3 import MP3
from waitress import serve
import aiohttp
app = Flask(__name__)
print("主人，我的工作目录是 "+dirpath+" 喵~")
#http://musicatrictl.akutan445.com:4949/songctl?id=
if platform.system() == "Windows":
    cmd("chcp 936")
with codecs.open(dirpath + "plays.json", encoding="utf-8", mode="r") as c:
    plays = json.loads(c.read())
with codecs.open(dirpath + "langpref.json", encoding="utf-8", mode="r") as c:
    langpref = json.loads(c.read())
with codecs.open(dirpath + "haogan.json", encoding='utf-8', mode='r') as f:
    userdata = json.loads(f.read())
with codecs.open(dirpath + "waifulist.txt", encoding="utf-8", mode="r") as f:
    waifulist = ast.literal_eval(f.read())

with codecs.open(dirpath + "blacklist.json", encoding="utf-8", mode="r") as f:
    blacklist = json.loads(f.read())
translations={}
for file in os.listdir(dirpath+"langfiles"):
    if file.find(".json") != -1:
        with codecs.open(dirpath + "langfiles/"+file, encoding="utf-8", mode="r") as f:
            translations[file]=json.loads(f.read())
ytdl_format_options = {
    'format': 'bestaudio/best',
    'outtmpl': dirpath+'ytdltemp/%(id)s.%(ext)s',
    'restrictfilenames': True,
    'noplaylist': True,
    'nocheckcertificate': True,
    'ignoreerrors': False,
    'logtostderr': False,
    'quiet': True,
    'no_warnings': True,
    'default_search': 'auto',
    'source_address': '0.0.0.0',  # bind to ipv4 since ipv6 addresses cause issues sometimes
#    'download_archive':dirpath+'ytdldownloads.txt'
}
if key["ytdlproxy"]:
    ytdl_format_options['proxy']=key['ytdlproxy']
print("我的名字是" + str(name) + "。谢谢主人给我起这么好听的名字！")
ytdl = yt_dlp.YoutubeDL(ytdl_format_options)



async def getsongdetails(id):
    if exists(dirpath + "./datacache/" + id):
        with codecs.open(dirpath + "./datacache/" + id, encoding='utf-8', mode='r') as f:
            return json.loads(f.read())
    else:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://music.163.com/api/song/detail/?id="+id+"&ids=%5B"+id+"%5D") as resp:
                results = await resp.json(content_type=None)
                results=results["songs"][0]
                with codecs.open(dirpath + "./datacache/" + id, encoding='utf-8', mode='w') as f:
                    f.write(json.dumps(results))
                return results
async def getsongname(id):
    a=await getsongdetails(id)
    return a["name"]
async def getsongartists(id):
    art = await getsongdetails(id)
    art=art["artists"]
    return artistslistpurifier(art)
async def searchsong(sn):
    async with aiohttp.ClientSession() as session:
        async with session.get(cloudmusicapiurl + "/search?keywords="+sn) as resp:
            results = await resp.json()
            if results["result"]['songCount'] == 0:
                return -1
            id = tuple(results["result"]["songs"])
            for i in id:
                with codecs.open(dirpath + "./datacache/" + str(i["id"]), encoding='utf-8', mode='w') as f:
                    f.write(json.dumps(i))
            return id

async def getplaylist(sn):
    async with aiohttp.ClientSession() as session:
        async with session.get(cloudmusicapiurl + "/playlist/track/all?id="+sn+"&limit=30") as resp:
            results = await resp.json()
            id =results["songs"]
            for i in id:
                with codecs.open(dirpath + "./datacache/" + str(i["id"]), encoding='utf-8', mode='w') as f:
                    i["artists"]=i.pop("ar")
                    i["album"] = i.pop("al")
                    f.write(json.dumps(i))

            nl = []
            for i in id:
                nl.append(str(i['id']))
            return nl
async def getalbum(sn):
    async with aiohttp.ClientSession() as session:
        async with session.get(cloudmusicapiurl + "/album?id="+sn) as resp:
            results = await resp.json()
            id =results["songs"]
            for i in id:
                with codecs.open(dirpath + "./datacache/" + str(i["id"]), encoding='utf-8', mode='w') as f:
                    i["artists"]=i.pop("ar")
                    i["album"] = i.pop("al")
                    f.write(json.dumps(i))

            nl = []
            for i in id:
                nl.append(str(i['id']))
            return nl

@app.route('/updatesongqueue', methods = ['POST'])
def updatesongqueue():
    guildid=int(request.json["guildid"])
    newqueuelist=request.json["newqueue"]
    newqueue={}
    oldqueue=queues[guildid]
    for song in newqueuelist:
        newqueue[song]=oldqueue[song]
    queues[guildid]=newqueue
    return "updated queue"+str(list(queues[guildid].keys()))
@app.route('/deletesong', methods = ['POST'])
def deletesong():
    guildid=int(request.json["guildid"])
    songname=request.json["songname"]
    queues[guildid].pop(songname)
    return "updated queue"+str(list(queues[guildid].keys()))
@app.route('/requestnewsong', methods = ['POST'])
async def requestnewsong():
    guildid=int(request.json["guildid"])
    a=request.json["songname"]
    guild=atri.get_guild(int(guildid))
    players[guildid] = discord.utils.get(atri.voice_clients, guild=guild)
    id = await getsongid(a)
    if type(id) == type(()):
        #目前还没有实现网页歌曲选择
        #默认选取第一手歌
        id=id[0]
    if id == -1:
        return "这是什么歌曲，亚托莉无法播放哦!"
    if not players[guildid].is_playing():
        if id:
            if not await dl163ali(id):
                return
            songandartname=str(await getsongartists(id)).replace("[", "").replace("]", "").replace("'", "") + "——" + str(await getsongname(id))
            cs[guildid] = songandartname
            file=dirpath + "./songcache/" + id + ".mp3"
            songduration[songandartname]=getmp3duration(file)
            players[guildid].play(discord.FFmpegPCMAudio(file), after=partial(ckqueue, guild))
            cstarttime[guildid]=int(time.time()*1000)
            add1play(id)
            return "正在播放："+songandartname
        else:
            vid = await getyt(a)
            cs[guildid] = vid[1]["title"]
            songduration[vid[1]["title"]]=vid[1]["duration"]

            players[guildid].play(vid[0], after=partial(ckqueue, guild))
            cstarttime[guildid]=int(time.time()*1000)

            add1play(vid[1]["url"])
            return "正在播放："+vid[1]["title"]
    else:
        if id:
            a=await dl163ali(id)
            if not a:
                return "暂时不支持vip歌曲，ご主人様ごめなさい！！"
            songandartname=str(await getsongartists(id)).replace("[", "").replace("]", "").replace("'", "") + "——" + str(await getsongname(id))
            file=dirpath + "./songcache/" + id + ".mp3"
            try:
                if songandartname in queues[guildid].keys():
                    songandartname=songandartname + "⠀⠀⠀" + str(random.randint(1000001, 9999999))
                    queues[guildid][songandartname] = [discord.FFmpegPCMAudio(file),{"url":id}]
                else:
                    queues[guildid][songandartname] = [discord.FFmpegPCMAudio(file),{"url":id}]
            except Exception as e:
                queues[guildid] = {}
                queues[guildid][songandartname] = [discord.FFmpegPCMAudio(file),{"url":id}]
            songduration[songandartname]=getmp3duration(file)

            return songandartname+"已添加到播放列表"
        else:
            song = await getyt(a)
            try:
                if song[1]["title"] in queues[guildid].keys():
                    newname=song[1]["title"] + "⠀⠀⠀" + str(random.randint(1000001, 9999999))
                    songduration[newname]=song[1]["duration"]
                    queues[guildid][newname] = song
                else:
                    queues[guildid][song[1]["title"]] = song
                    songduration[song[1]["title"]]=song[1]["duration"]
            except:
                queues[guildid] = {}
                queues[guildid][song[1]["title"]] = song
                songduration[song[1]["title"]]=song[1]["duration"]
            return song[1]["title"]+"已添加到播放列表"
    return "ok"
@app.route('/getcurrentsong', methods = ['GET'])
def getcurrentsong():
    try:
        args = request.args.to_dict()
        a={"songname":cs[int(args["id"])],"duration":songduration[cs[int(args["id"])]],"starttime":cstarttime[int(args["id"])],"playing":players[int(args["id"])].is_playing()}
        return json.dumps(a)
    except:
        return json.dumps({})
@app.route('/getcurrentqueue', methods = ['GET'])
def getcurrentqueue():
    args = request.args.to_dict()
    try:
        return json.dumps(list(queues[int(args["id"])].keys()))
    except:
        return json.dumps([])
@app.route('/changesongstate', methods = ['POST'])
def changesongstate():
    if request.json["action"] == "next":
        players[int(request.json["guildid"])] = discord.utils.get(atri.voice_clients, guild=atri.get_guild(int(request.json["guildid"])))
        if players[int(request.json["guildid"])]:
            try:
                if queues[int(request.json["guildid"])]:
                    players[int(request.json["guildid"])].stop()
                    id = next(iter(queues[int(request.json["guildid"])]))
                    players[int(request.json["guildid"])] = discord.utils.get(atri.voice_clients, guild=atri.get_guild(int(request.json["guildid"])))
                    badidea = id.find("⠀⠀⠀")
                    if badidea != -1:
                        id = id[:badidea]
                    cs[int(request.json["guildid"])] = id
                else:
                    players[int(request.json["guildid"])].stop()
            except:
                print(traceback.format_exc())
                players[int(request.json["guildid"])].stop()
        else:
            return "no"
    elif request.json["action"] == "pause":
        players[int(request.json["guildid"])] = discord.utils.get(atri.voice_clients, guild=atri.get_guild(int(request.json["guildid"])))
        if players[int(request.json["guildid"])]:
            if players[int(request.json["guildid"])].is_playing():
                players[int(request.json["guildid"])].pause()
                pausesong(int(request.json["guildid"]))
            else:
                players[int(request.json["guildid"])].resume()
                cstarttime[int(request.json["guildid"])]=cstarttime[int(request.json["guildid"])]+pausesong(int(request.json["guildid"]))
        else:
            return "no"
    else:
        return "这是什么命令，亚托莉看不懂。亚托莉去问一下夏生先生！"
    return "ok"
@app.route('/songctl',methods = ['GET'])
def songctl():
    return send_file(dirpath+"website/songctl.html")
@app.route('/<path:path>')
def send_report(path):
    return send_from_directory('website', path)

def getmp3duration(file):
    audio = MP3(file)
    return audio.info.length

def mutisearch(s,t):
    for i in t:
        if s.find(i) != -1:
            return True
    return False

def replacetrans(message, userid, *replace):
    userid = str(userid)
    if userid not in langpref:
        langpref[userid] = "zh.json"
        default_message = "You have not set a language, defaulting to Chinese Simplified. You can set a language with " + name[0] + "langset."
        translation = translations[langpref[userid]][message]
        if replace:
            if len(replace) > 1:
                chosen_message = random.choice(translation)
                return default_message + "\n" + chosen_message.replace("%a", replace[0]) if "%a" in chosen_message else default_message + "\n" + chosen_message
            return default_message + "\n" + translation.replace("%a", replace[0]) if "%a" in translation else default_message + "\n" + translation
        return default_message + "\n" + translation

    translation = translations[langpref[userid]][message]
    if replace:
        if len(replace) > 1:
            chosen_message = random.choice(translation)
            return chosen_message.replace("%a", replace[0]) if "%a" in chosen_message else chosen_message
        return translation.replace("%a", replace[0]) if "%a" in translation else translation
    return translation


def testbyn(sn):
    sl = ["youtube.com", "youtu.be", "bilibili.com", "nicovideo.jp", "nico.ms", "b23.tv"]
    return any(sub in sn for sub in sl)


class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, ):
        super().__init__(source)
        # YTDL info dicts (data) have other useful information you might want
        # https://github.com/rg3/youtube-dl/blob/master/README.md
    def __getitem__(self, item: str):
        """Allows us to access attributes similar to a dict.
        This is only useful when you are NOT downloading.
        """
        return self.__getattribute__(item)
    @classmethod
    async def create_source(cls, search: str, ):

        if mutisearch(search, ["b23.tv","bili"]):
            site="bili"
        elif mutisearch(search, ["nicovideo.jp", "nico.ms"]):
            site="nico"
        else:
            site="yt"

        loop = asyncio.get_event_loop()
        to_run = partial(ytdl.extract_info, url=search)
        data = await loop.run_in_executor(None, to_run)
        if key["devmode"]:
            with open(dirpath + "./ytdltemp/" + data['id'] + ".info.json", "w") as f:
                f.write(json.dumps(data, sort_keys=True, indent=4))

        if 'entries' in data:
            lista=[]
            for d in data['entries']:
                source = ytdl.prepare_filename(d)
                if site=="bili":
                    d["url"] = "https://www.bilibili.com/video/"+d["webpage_url_basename"]
                elif site=="nico":
                    d["url"] = "https://www.nicovideo.jp/watch/" + d["webpage_url_basename"]
                else:
                    d["url"] = d["webpage_url"]
                lista.append([discord.FFmpegPCMAudio(source),d])
            return lista

        else:

            source = ytdl.prepare_filename(data)
            if site=="bili":
                data["url"] = "https://www.bilibili.com/video/" + data["webpage_url_basename"]
            elif site=="nico":
                data["url"] = "https://www.nicovideo.jp/watch/" + data["webpage_url_basename"]
            else:
                data["url"]=data["webpage_url"]
            return (discord.FFmpegPCMAudio(source),data)

async def getyt(url):
    a = await YTDLSource.create_source(url)
    return a

def add1play(id):
    try:
        plays[id] = plays[id] + 1
    except:
        plays[id] = 1

async def playt(ctx, vid):
    # “a” is the video url
    cs[ctx.guild.id] = vid[1]["title"]
    songduration[vid[1]["title"]]=vid[1]["duration"]
    players[ctx.guild.id].play(vid[0], after=partial(ckqueue, ctx.guild))
    cstarttime[ctx.guild.id]=int(time.time()*1000)
    await ctx.send(    replacetrans("now_playing",ctx.author.id,vid[1]["title"]) )
    await ctx.send(replacetrans("show_web_address_user",ctx.author.id,key["songctladdr"]+str(ctx.guild.id)))
    add1play(vid[1]["url"])

async def addtoqueueyt(ctx, song):
    if type(song) == type([]):
        adding[ctx.guild.id] = True
        songs=song
        songaddtext=""
        for song in songs:
            if adding[ctx.guild.id]:
                try:
                    if song[1]["title"] in queues[ctx.guild.id].keys():
                        queues[ctx.guild.id][song[1]["title"] + "⠀⠀⠀" + str(random.randint(1000001, 9999999))] = song
                    else:
                        queues[ctx.guild.id][song[1]["title"]] = song
                except:
                    queues[ctx.guild.id] = {}
                    queues[ctx.guild.id][song[1]["title"]] = song
                songduration[song[1]["title"]]=song[1]["duration"]

                songaddtext=songaddtext+song[1]["title"] + "\n"

            else:
                await ctx.send(replacetrans("stopadding",ctx.author.id))
                break
        await ctx.send(replacetrans("added_to_playlist",ctx.author.id, songaddtext))
    else:
        try:
            if song[1]["title"] in queues[ctx.guild.id].keys():
                queues[ctx.guild.id][song[1]["title"] + "⠀⠀⠀" + str(random.randint(1000001, 9999999))] = song
            else:
                queues[ctx.guild.id][song[1]["title"]] = song
        except:
            queues[ctx.guild.id] = {}
            queues[ctx.guild.id][song[1]["title"]] = song
        songduration[song[1]["title"]]=song[1]["duration"]

        await ctx.send(replacetrans("added_to_playlist",ctx.author.id, song[1]["title"]))

async def addtoqueue163(ctx, id):
    if id==-1:
        #canceled
        return
    if type(id) == type([]):
        adding[ctx.guild.id] = True
        for i in id:
            if adding[ctx.guild.id]:
                if not await dl163ali(i):
                    await ctx.send(replacetrans("error_vip_not_supported",ctx.author.id))
                    continue
                else:
                    songname=str(await getsongartists(i)).replace("[", "").replace("]", "").replace("'","") + "——" + str(await getsongname(i))
                    file=dirpath + "./songcache/" + i + ".mp3"
                    try:
                        songduration[songname]=getmp3duration(file)
                        try:
                            if songname in queues[ctx.guild.id].keys():
                                songname=songname + "⠀⠀⠀" + str(random.randint(1000001, 9999999))
                                queues[ctx.guild.id][songname] = [discord.FFmpegPCMAudio(file),{"url":i}]
                            else:
                                queues[ctx.guild.id][songname] = [discord.FFmpegPCMAudio(file),{"url":i}]
                        except Exception as e:
                            #print(e)
                            queues[ctx.guild.id] = {}
                            #print("reset queue")
                            queues[ctx.guild.id][songname] = [discord.FFmpegPCMAudio(file),{"url":i}]

                        await ctx.send(replacetrans("added_to_playlist",ctx.author.id,songname))
                    except:
                        await ctx.send(replacetrans("error_vip_not_supported",ctx.author.id))
            else:
                await ctx.send(replacetrans("stopading",ctx.author.id))
                break
    else:
        a=await dl163ali(id)
        if not a:
            await ctx.send(replacetrans("error_vip_not_supported",ctx.author.id))
            return
        songname=str(await getsongartists(id)).replace("[", "").replace("]", "").replace("'", "") + "——" + str(await getsongname(id))
        file=dirpath + "./songcache/" + id + ".mp3"
        songduration[songname]=getmp3duration(file)
        try:
            if songname in queues[ctx.guild.id].keys():
                queues[ctx.guild.id][songname + "⠀⠀⠀" + str(random.randint(1000001, 9999999))] = [discord.FFmpegPCMAudio(file),{"url":id}]
            else:
                queues[ctx.guild.id][songname] = [discord.FFmpegPCMAudio(file),{"url":id}]
        except Exception as e:
            queues[ctx.guild.id] = {}
            queues[ctx.guild.id][songname] =  [discord.FFmpegPCMAudio(file),{"url":id}]
        await ctx.send(replacetrans("added_to_playlist",ctx.author.id,songname))

def ckqueue(guild, uselessd, uselessd2=None):
    try:
        if guild.id not in queues or not queues[guild.id]:
            cs[guild.id] = False
            return
        id = next(iter(queues[guild.id]))
        song = queues[guild.id].pop(id)
        players[guild.id] = discord.utils.get(atri.voice_clients, guild=guild)
        cs[guild.id] = id
        players[guild.id].play(song[0], after=partial(ckqueue, guild))
        cstarttime[guild.id]=int(time.time()*1000)
        add1play(song[1]["url"])
    except Exception as e:
        cs[guild.id] = False

openai.api_key = key["gptkey"]
queues = {}
adding = {}
cs = {}
players = {}
waifucd = {}
intents = discord.Intents.all()
atri = commands.AutoShardedBot(command_prefix=name, intents=intents,help_command=None)
scm={}
songduration={}
cstarttime={}
pausetime={}
@atri.event
async def on_member_update(before, after):
    if str(before.guild.id) == "937939784494612570":
        if str(before.id) in waifulist:
            if before.status == after.status:
                pass
            else:
                if before.id in waifucd.keys():
                    if waifucd[str(before.id)] + 1800 > round(time.time()):
                        return
                if str(after.status) == "online" or (str(after.status) == "away" and str(before.status) == "offline"):
                    await after.send(replacetrans("master_arrive_home_catgirl",before.id,str(after.name),True))
                    waifucd[str(before.id)] = round(time.time())
                elif str(after.status) == "offline" or str(after.status) == "away":
                    await after.send(replacetrans("choice_master_leave",before.id,str(after.name),True))
                    waifucd[str(before.id)] = round(time.time())
                else:
                    pass

class chatgpt():
    def __init__(self, channel):
        self.messages=[]
        self.channel=channel
    async def loadmessages(self,messageurl):
        r = requests.get(messageurl)
        self.messages=ast.literal_eval(r.text.replace("\n","").replace("'","\""))
        await self.channel.send("ok")
    async def savemessages(self):
        with codecs.open("savechat.txt",encoding='utf-8', mode='w') as f:
            f.write(str(self.messages))
        with open("savechat.txt", "rb") as file:
            await self.channel.send("聊天文件:", file=discord.File(file, "savechat.txt"))
            return
    async def loadpreset(self,preset):
        with codecs.open(dirpath+"/gptprompt/"+preset+".txt",encoding='utf-8', mode='r') as f:
            prompt=f.read()
        self.messages.append({"role": "user", "content":  prompt})
        await self.channel.send("ok")
    async def gererateresponse(self,newmessage):
        try:
            self.messages.append({"role": "user", "content":  newmessage})
            completion =  await asyncio.to_thread( openai.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=self.messages,

            )
            chat_response = completion.choices[0].message.content
            #chat_response = "testing"
            await self.channel.send(chat_response)
            self.messages.append({"role": "assistant", "content": chat_response})
        except:
            if key['devmode']:
                await self.channel.send("亚托莉，坏掉了！")
                with codecs.open(dirpath + "./err.txt", encoding='utf-8', mode='w') as file:
                    file.write(str(traceback.format_exc()))
                # send file to Discord in message
                with open("err.txt", "rb") as file:
                    await self.channel.send("错误文件：", file=discord.File(file, "err.txt"))
                return
            else:
                await self.channel.send("ChatGPT API调用失败，请重试。")

@atri.event
async def on_message(message):
    if key["gptkey"]:
        if message.author == atri.user:
            return
        askgpt=False
        atri_id = f"{atri.user.id}".split("#")[0]
        if message.reference:
            try:
                replied_message = await message.channel.fetch_message(message.reference.message_id)
                if replied_message.author == atri.user:
                    askgpt = True
            except discord.NotFound:
                pass
        if not askgpt and not message.content.startswith(f"<@{atri_id}>"):
            if not (str(message.author.id) in blacklist.keys()):
                await atri.process_commands(message)
        else:
            cleanmessage=message.content.replace(f"<@{atri_id}>", "").strip()
            if message.guild.id not in scm.keys():
                scm[message.guild.id]=chatgpt(message.channel)
            gpt=scm[message.guild.id]
            if cleanmessage[:4]=="保存对话":
                await gpt.savemessages()
            elif cleanmessage[:4]=="加载对话":
                await gpt.loadmessages(cleanmessage[4:].replace(" ",""))
            elif cleanmessage[:4]=="加载角色":
                await gpt.loadpreset(cleanmessage[4:].replace(" ",""))
            else:
                await gpt.gererateresponse(cleanmessage)
    else:

        if not (str(message.author.id) in blacklist.keys()):
            await atri.process_commands(message)

@atri.event
async def on_ready():
    print("主人我上线啦(｡･ω･｡)ﾉ♡")
    print("主人我目前加入了"+str(len(atri.guilds))+"个服务器哦")
    writeplays.start()
    await atri.change_presence(activity=discord.Activity(type=discord.ActivityType.listening,name="主人的命令||" + name[0] + "play <歌曲>||支持网易云，哔哩哔哩，youtube，ニコニコ"))

async def getsongid(sn):
    b = sn.find("song?id=")
    id = ""
    if b == -1:
        id = sn
        try:
            int(id)
        except:
            if testbyn(sn):
                return False
            listtextl = sn.find("lbum?id=")
            if listtextl != -1:

                # slow
                sn = sn[listtextl + 8:]
                return await getalbum(sn)
            listtextl = sn.find("list?id=")
            if listtextl != -1:
                # slow
                sn=sn[listtextl + 8:]
                return await getplaylist(sn)
            # reload(search163)
            # slow
            #id = await asyncio.get_event_loop().run_in_executor(None, search163.get_id_and_cache_data, sn)
            # searcg 1


            id= await searchsong(sn)
            #lbum?id=


    else:
        n = sn[b + 8:]
        if n.find("&") == -1:
            id = n
        else:
            id = n[:n.find("&")]
    return id

async def dl163ali(id):
    if exists(dirpath + "./songcache/" + id + ".mp3"):
        return True
    #/song/url/v1?id=33894312&level=exhigh
    async with aiohttp.ClientSession() as session:
        async with session.get(cloudmusicapiurl+"/song/url/v1?id="+id+"&level=higher&cookie="+cookie) as resp:
            #print(cloudmusicapiurl+"/song/url/v1?id="+id+"&level=exhigh&cookie="+cookie)
            results=await resp.json()
            async with aiohttp.ClientSession() as session:
                async with session.get(results["data"][0]["url"]) as resp:
                    if resp.status == 200:
                        f = await aiofiles.open(dirpath + "./songcache/" + id + ".mp3", "wb")
                        await f.write(await resp.read())
                        await f.close()
                        return True
@atri.command(aliases=["封"])
async def ban(ctx, id, reason=" "):
    if str(ctx.message.author.id) == "834651231871434752":
        blacklist[str(id)] = reason
        await ctx.send("ok")


@atri.command(aliases=["解封"])
async def unban(ctx, id):
    if str(ctx.message.author.id) == "834651231871434752":
        blacklist.pop(str(id))
        await ctx.send(replacetrans("master_is_so_kind",ctx.author.id))

@atri.command()
async def langset(ctx, *lang):
    global langpref
    if not lang:
        await ctx.send("avaliable languages:"+str(os.listdir(dirpath+"langfiles")))
    else:
        if lang[0] in os.listdir(dirpath+"langfiles") or lang[0]+".json" in os.listdir(dirpath+"langfiles"):
            langpref[str(ctx.author.id)]=lang[0].replace(".json","")+".json"
        else:
            await ctx.send("avaliable languages:"+str(os.listdir(dirpath+"langfiles")))

@atri.command()
async def spelling(ctx):
    def check(m):
        return m.channel == ctx.channel

    if ctx.author.voice:
        pass
    else:
        await ctx.send(replacetrans("spelling_please_connect_to_voice",ctx.author.id))
        return
    await ctx.send(replacetrans("spelling_test_start_add",ctx.author.id))
    returnlist = await atri.wait_for('message', check=check)

    try:
        splist = ast.literal_eval(returnlist.content)
        random.shuffle(splist)
        await ctx.send(replacetrans("spelling_start",ctx.author.id))
        await returnlist.delete()
        if ctx.voice_client:
            if ctx.voice_client.is_connected:
                await ctx.voice_client.disconnect()
        await ctx.message.author.voice.channel.connect()
        players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
        reviewlist = []
        for w in splist:
            while (1):
                if exists(dirpath + "./temp" + str(ctx.guild.id) + ".mp3"):
                    os.remove(dirpath + "./temp" + str(ctx.guild.id) + ".mp3")
                word = gTTS(text=w, lang="en", slow=True)
                word.save(dirpath + "./temp" + str(ctx.guild.id) + ".mp3")
                try:
                    players[ctx.guild.id].play(discord.FFmpegPCMAudio(dirpath + "./temp" + str(ctx.guild.id) + ".mp3"))
                    cstarttime[ctx.guild.id]=int(time.time()*1000)
                except:
                    pass
                await ctx.send(replacetrans("spelling_test_please_respond",ctx.author.id))
                ansobj = await atri.wait_for('message', check=check)
                #print("waiting")
                ans = ansobj.content
                if ans.lower() == w.lower():
                    #print("correct")
                    await ctx.send(replacetrans("spelling_correct",ctx.author.id))
                    break
                elif ans == "r":
                    #print("retry")
                    pass
                else:
                    #print("wrong")
                    await ctx.send(replacetrans("spelling_wrong",ctx.author.id,w))
                    reviewlist.append(w)
                    while (1):
                        players[ctx.guild.id].play(discord.FFmpegPCMAudio(dirpath + "./temp" + str(ctx.guild.id) + ".mp3"))
                        await ctx.send(replacetrans("spelling_wrong_reenter",ctx.author.id,w))
                        ansobj = await atri.wait_for('message', check=check)
                        ans = ansobj.content
                        if ans.lower() == w.lower():
                            await ctx.send(replacetrans("spelling_wrong_correct",ctx.author.id))
                            break
                        elif ans == "r":
                            pass
                        else:
                            await ctx.send(replacetrans("spelling_wrong_wrong",ctx.author.id))
                    break
        await ctx.send(str((len(splist) - len(reviewlist) / len(splist))) + "分")
        await ctx.send(replacetrans("spelling_learned_words",ctx.author.id,splist))
        await ctx.send(replacetrans("spelling_failed_words",ctx.author.id,reviewlist))

    except:
        await ctx.send(replacetrans("error_atri_broken",ctx.author.id))
        with codecs.open(dirpath + "./err.txt", encoding='utf-8', mode='w') as file:
            file.write(str(traceback.format_exc()))
        # send file to Discord in message
        with open("err.txt", "rb") as file:
            await ctx.send("错误文件：", file=discord.File(file, "err.txt"))
        return

@atri.command(aliases=["すき", "最喜欢了", "喜欢", "爱"])
async def suki(ctx, *v):
    user_id = str(ctx.author.id)

    def reset_daily_limits():
        userdata[user_id]["dailylimits"] = {"pat": 0, "suki": 0, "727": 0}

    def update_haogandu(value):
        userdata[user_id]["haogandu"] += value

    async def send_message(key, value_change):
        await ctx.send(replacetrans(key, user_id))
        update_haogandu(value_change)

    if not v or not v[0].isdigit():
        await ctx.send(replacetrans("suki_missing_args", user_id))
        return

    user_data = userdata.get(user_id, {})
    if user_data.get("dailylimittime") != str(date.today()):
        reset_daily_limits()

    haogandu = user_data.get("haogandu", 0)
    daily_limit = user_data.get("dailylimits", {}).get("suki", 0)

    if haogandu >= -20:
        if daily_limit < 10:
            if int(v[0]) > 5:
                await send_message("suki_love", 1)
            else:
                await send_message("suki_kirai", -5 if haogandu <= 500 else -400)
        else:
            await send_message("suki_atricon", -10 if haogandu <= 500 else 10)
    else:
        await send_message("suki_report_police", -20)

    userdata[user_id]["dailylimits"]["suki"] = daily_limit + 1


@atri.command(aliases=["摸头"])
async def pat(ctx):
    try:
        if userdata[str(ctx.author.id)]["dailylimittime"] != str(date.today()):
            userdata[str(ctx.author.id)]["dailylimits"]["pat"] = 0
            userdata[str(ctx.author.id)]["dailylimits"]["suki"] = 0
            userdata[str(ctx.author.id)]["dailylimits"]["727"] = 0
        if not userdata[str(ctx.author.id)]["haogandu"] < 0:
            if userdata[str(ctx.author.id)]["dailylimits"]["pat"] > 5:
                await ctx.send("呜~头发都被弄乱了，主人能不能休息一下吗？")
            if userdata[str(ctx.author.id)]["dailylimits"]["pat"] > 10:
                await ctx.send("主人怎么一天到晚就只会摸小萝莉的头。难道是萝莉控？？好恶心！")
                userdata[str(ctx.author.id)]["haogandu"] = userdata[str(ctx.author.id)]["haogandu"] - 10
            else:
                userdata[str(ctx.author.id)]["haogandu"] = userdata[str(ctx.author.id)]["haogandu"] + random.randint(0,
                                                                                                                    5)
                await ctx.send("好舒服~最喜欢被主人摸头啦")
            userdata[str(ctx.author.id)]["dailylimits"]["pat"] = userdata[str(ctx.author.id)]["dailylimits"]["pat"] + 1
            userdata[str(ctx.author.id)]["dailylimittime"] = str(date.today())
        else:
            await ctx.send("救命！！这边有个变态在骚扰我！")
    except:
        # print(traceback.format_exc())
        userdata[str(ctx.author.id)] = {}
        userdata[str(ctx.author.id)]["dailylimits"] = {}
        userdata[str(ctx.author.id)]["dailylimits"]["pat"] = 1
        userdata[str(ctx.author.id)]["dailylimittime"] = str(date.today())
        userdata[str(ctx.author.id)]["dailylimits"]["suki"] = 0
        userdata[str(ctx.author.id)]["dailylimits"]["727"] = 0
        userdata[str(ctx.author.id)]["haogandu"] = random.randint(0, 5)
        await ctx.send("好舒服~最喜欢被主人摸头啦")

@atri.command(aliases=["誓约"])
async def marry(ctx):
    if userdata[str(ctx.author.id)]["haogandu"] < 100:
        await ctx.send("No no we are friends! friends!")
    else:
        await ctx.send("ok")
        await ctx.author.send("主人来这里！https://discord.gg/x2VMR2uAju")
        waifulist.append(str(ctx.author.id))

@atri.command(aliases=["排行榜"])
async def rankings(ctx):
    songtable = PrettyTable()
    songtable.field_names = ["排名","播放次数", "歌手/歌曲名"]
    songtable.align = 'l'
    songtable.set_style(PLAIN_COLUMNS)
    ct = 1
    msg = "```!全dc亚托莉放的最多的歌曲前二 十!\n"
    for id in sorted(plays, key=plays.get, reverse=True)[:20]:
        try:
            int(id)
            songtable.add_row([ct,  plays[id],str(await getsongartists(id)) + "——" + str(await getsongname(id))])
            #msg = msg + str(ct) + ".  " +str(await getsongartists(id)) + "——" + str(await getsongname(id))+ " || " + str(plays[id]) + "次播放。\n"
            ct = ct + 1
        except:
            #msg = msg + str(ct) + ".  " +id+ " || " + str(plays[id]) + "次播放。\n"
            songtable.add_row([ct, plays[id], id ])
            ct = ct + 1



    await ctx.send(msg+str(songtable)+"```")
def artistslistpurifier(j):
    nl=[]
    for i in j:
        nl.append(i['name'])
    return "，".join(nl)
async def songchoice(ctx,xuanze):
    def check(m):
        return m.channel == ctx.channel
    await ctx.send(replacetrans("select_song",str(ctx.author.id)))
    msg=""
    for x in range(len(xuanze)):
        cr=xuanze[x]
        msg = msg + str(x) + ".  " + str(artistslistpurifier(cr['artists'])) + "——" + cr["name"] +"\n"
    await ctx.send(msg)
    selection = await atri.wait_for('message',check=check )
    selection=selection.content
    try:
        return str(xuanze[int(selection)-1]["id"])
    except:
        await ctx.send(replacetrans("select_song_cancel",str(ctx.author.id)))
        return -1



@atri.command(aliases=["播放", "queue", "播放列表"])
async def play(ctx, *a):

    try:
        if a:
            if ctx.author.voice:
                players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
                if not players[ctx.guild.id]:
                    await ctx.message.author.voice.channel.connect()
                    players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
                a =" ".join(a)
                id = await getsongid(a)
                if id == -1:
                    return
                if not players[ctx.guild.id].is_playing():
                    if id:
                        if type(id) == type([]):
                            fid = id.pop(0)
                            await play163(ctx, fid)
                            await addtoqueue163(ctx, id)
                        elif type(id) == type(()):
                            await play163(ctx,await songchoice(ctx,id))
                        else:
                            await play163(ctx, id)
                    else:
                        song = await getyt(a)
                        if type(song) == type([]):
                            fid = song.pop(0)
                            await playt(ctx, fid)
                            await addtoqueueyt(ctx, song)
                        else:
                            await playt(ctx, song)
                else:
                    if id:
                        #returned a search reult list

                        if type(id)==type(()):
                            await addtoqueue163(ctx, await songchoice(ctx,id))
                        else:
                            await addtoqueue163(ctx, id)
                    else:
                        song = await getyt(a)
                        await addtoqueueyt(ctx,song)
            else:
                await ctx.send(replacetrans("error_not_connected",str(ctx.author.id)))
        else:
            if not key["devmode"]:
                await ctx.send(replacetrans("playlist",str(ctx.author.id)))
                try:
                    if queues[ctx.guild.id].keys() == []:
                        await ctx.send(replacetrans("no_songs",str(ctx.author.id)))
                    else:
                        bbbb = ""
                        for aaa in queues[ctx.guild.id].keys():
                            badidea = aaa.find("⠀⠀⠀")
                            if badidea != -1:
                                aaa = aaa[:badidea]

                            bbbb = bbbb + "\n" + aaa
                        await ctx.send(bbbb)
                except Exception as e:
                    #print(e)
                    await ctx.send(replacetrans("no_songs",str(ctx.author.id)))
            else:
                await ctx.send(str(queues))
                await ctx.send(replacetrans("playlist",str(ctx.author.id)))
                try:
                    if queues[ctx.guild.id].keys() == []:
                        await ctx.send(replacetrans("no_songs",str(ctx.author.id)))
                    else:
                        bbbb = ""
                        for aaa in queues[ctx.guild.id].keys():
                            badidea = aaa.find("⠀⠀⠀")
                            if badidea != -1:
                                aaa = aaa[:badidea]

                            bbbb = bbbb + "\n" + aaa
                        await ctx.send(bbbb)
                except Exception as e:
                    #print(e)
                    await ctx.send(replacetrans("no_songs",str(ctx.author.id)))
                #print(str(queues))
    except Exception as e:

        if key["devmode"]:
            await ctx.send(replacetrans("error_atri_broken",str(ctx.author.id)))
            with codecs.open(dirpath + "./err.txt", encoding='utf-8', mode='w') as file:
                file.write(str(traceback.format_exc()))
            # send file to Discord in message
            with open("err.txt", "rb") as file:
                await ctx.send( replacetrans("error_traceback",str(ctx.author.id)), file=discord.File(file, "err.txt"))
        else:
            await ctx.send("播放失败")

async def play163(ctx, id):
    if id==-1:
        #canceled
        return
    if not await dl163ali(id):  # 调用await dl163ali，如果歌曲可以下载会下载歌曲，不可以的话会返回 False 所以下面不需要调用
        await ctx.send(replacetrans("error_vip_not_supported",ctx.author.id))
        return
    songname=str(await getsongartists(id)) + "——" + str(await getsongname(id))
    cs[ctx.guild.id] = songname
    file=dirpath + "./songcache/" + id + ".mp3"
    songduration[songname]=getmp3duration(file)
    players[ctx.guild.id].play(discord.FFmpegPCMAudio(file), after=partial(ckqueue, ctx.guild))
    cstarttime[ctx.guild.id]=int(time.time()*1000)
    await ctx.send(replacetrans("now_playing",ctx.author.id,songname))
    await ctx.send(replacetrans("show_web_address_user",ctx.author.id,key["songctladdr"]+str(ctx.guild.id)))
    add1play(id)
def pausesong(guildid):
    if guildid in pausetime.keys():
        timea=pausetime.pop(guildid)
        return int(time.time()*1000)-timea
    else:
        pausetime[guildid]=int(time.time()*1000)
        return -1

@atri.command(aliases=["断开"])
async def disconnect(ctx, *a):
    players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
    if players[ctx.guild.id]:
        await players[ctx.guild.id].disconnect()
        await ctx.send(replacetrans("bye",ctx.author.id))
    else:
        await ctx.send(replacetrans("bye_mad",ctx.author.id))

@atri.command(aliases=["连接"])
async def connect(ctx, *a):
    if ctx.author.voice:
        await ctx.message.author.voice.channel.connect()
        players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
        await ctx.send(replacetrans("connect",ctx.author.id))
        await ctx.send(replacetrans("show_web_address_user",ctx.author.id,key["songctladdr"]+str(ctx.guild.id)))
    else:
        await ctx.send(replacetrans("error_not_connected",ctx.author.id))

@atri.command(aliases=["数数"])
async def count(ctx, *v):
    if ctx.author.voice:
        await ctx.message.author.voice.channel.connect()
        players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
        players[ctx.guild.id].play(discord.FFmpegPCMAudio(dirpath + "count.mp3"))
    else:
        await ctx.send(replacetrans("error_not_connected",ctx.author.id))

@atri.command(aliases=["劈瓜", "gua"])
async def pigua(ctx, *v):
    if ctx.author.voice:
        await ctx.message.author.voice.channel.connect()
        players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
        players[ctx.guild.id].play(discord.FFmpegPCMAudio(dirpath + "gua.mp3"))
    else:
        await ctx.send(replacetrans("killer",ctx.author.id))

@atri.command(aliases=["gomenasai", "对不起", "本当にごめんなさい", "ほんどにごめなさい"])
async def sorry(ctx, *v):
    "特殊需求652615081682796549"
    try:
        if ctx.message.author.voice:
            for m in ctx.message.author.voice.channel.members:
                if m.id == 652615081682796549:
                    # await ctx.send("目标频道里有可怕的coffee，拒绝执行。红豆泥！私密马赛~~~")
                    raise RuntimeError("当前频道里有可怕的coffee，拒绝执行。红豆泥！私密马赛~~~")
        if ctx.voice_client:
            if ctx.voice_client.is_connected:
                for m in ctx.voice_client.channel.members:
                    if m.id == 652615081682796549:
                        # await ctx.send("当前频道里有可怕的coffee，拒绝执行。红豆泥！私密马赛~~~")
                        raise RuntimeError("当前频道里有可怕的coffee，拒绝执行。红豆泥！私密马赛~~~")
        pass
    except Exception as e:
        await ctx.send("亚托莉炸了 https://cdn.discordapp.com/attachments/1123003419137941626/1168483140919103528/ev012b3.png")
        with open("err.txt", "w") as file:
            file.write(str(traceback.format_exc()))
        # send file to Discord in message
        with open("err.txt", "rb") as file:
            await ctx.send("错误文件：", file=discord.File(file, "err.txt"))
        return
    if ctx.author.voice:
        if ctx.voice_client:
            if ctx.voice_client.is_connected:
                await ctx.voice_client.disconnect()
        await ctx.message.author.voice.channel.connect()
        players[ctx.guild] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
        players[ctx.guild].play(discord.FFmpegPCMAudio(dirpath + "honndonigomenasai.mp3"))
        await asyncio.sleep(5)
        await ctx.voice_client.disconnect()
    else:
        await ctx.send("红豆泥！私密马赛~~~")

@atri.command(aliases=["暂停","resume","继续"])
async def pause(ctx, *a):
    players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
    if players[ctx.guild.id]:
        if players[ctx.guild.id].is_playing():
            players[ctx.guild.id].pause()
            pausesong(int(ctx.guild.id))
        else:
            players[int(ctx.guild.id)].resume()
            cstarttime[int(ctx.guild.id)]=cstarttime[ctx.guild.id]+pausesong(ctx.guild.id)
    else:
        await ctx.send(replacetrans("not_connected_pause",ctx.author.id))

@atri.command(aliases=["停"])
async def stop(ctx, *a):
    players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
    if players[ctx.guild.id]:
        players[ctx.guild.id].stop()
        await ctx.send(replacetrans("stop",ctx.author.id))
    else:
        await ctx.send(replacetrans("not_connected_stop",ctx.author.id))

@atri.command(aliases=["跳过", "下一首"])
async def skip(ctx, a=1):
    players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
    if players[ctx.guild.id]:
        try:
            if queues[ctx.guild.id]:
                try:
                    for x in range(int(a)):
                        players[ctx.guild.id].stop()
                        id = next(iter(queues[ctx.guild.id]))
                        players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
                    badidea = id.find("⠀⠀⠀")
                    if badidea != -1:
                        id = id[:badidea]
                    await ctx.send(replacetrans("now_playing",ctx.author.id,id))
                    cs[ctx.guild.id] = id
                except:
                    players[ctx.guild.id].stop()
                    id = next(iter(queues[ctx.guild.id]))
                    players[ctx.guild.id] = discord.utils.get(atri.voice_clients, guild=ctx.guild)
                    badidea = id.find("⠀⠀⠀")
                    if badidea != -1:
                        id = id[:badidea]
                    await ctx.send(replacetrans("now_playing",ctx.author.id,id))
                    cs[ctx.guild.id] = id
            else:
                players[ctx.guild.id].stop()
        except:
            players[ctx.guild.id].stop()
    else:
        await ctx.send(replacetrans("not_conneted_skip",ctx.author.id))

@atri.command()
async def level(ctx, *a):
    await ctx.send(userdata[str(ctx.author.id)]["haogandu"])

@atri.command()
async def stopadding(ctx):
    adding[ctx.guild.id] = False
    await ctx.send("ok")
@atri.command()
async def help(ctx):
    await ctx.send("""
    A. """+name[0]+"""play <YouTube/Bilibili video/playlist url,NetEase Cloud Music Link/Playlist/Search Keyword >
        Join a voice channel and play the specified song. If a search keyword is provided, it will search for the keyword on NetEase Cloud Music and play the first result.
        If music is already playing, add the song to the playlist.
        If no specific song is specified, return the playlist.

    B. """+name[0]+"""pause

        Pause the current song.

    C. """+name[0]+"""continue

        Resume playing if the current song is paused.

    D. """+name[0]+"""currentsong

        return the name of the song that is currently playing

    G. """+name[0]+"""skip

        Skip the current song

    H. """+name[0]+"""stop

        Stop playback

    I. """+name[0]+"""connect

        Join to the voice channel

    J. """+name[0]+"""disconnect

        leave the voice channel
        """)
@atri.command()
async def clearqueue(ctx):
    await ctx.send("ok")
    queues[ctx.guild.id] = {}

@atri.command(aliases=["当前歌曲", "cs"])
async def currentsong(ctx, *a):
    if cs[ctx.guild.id]:
        await ctx.send(replacetrans("now_playing",ctx.author.id,cs[ctx.guild.id]))

@atri.command()
async def fix(ctx):
    user=await atri.fetch_user(834651231871434752)
    efn=dirpath + "./err"+str(round(time.time()))+".txt"
    with codecs.open(efn, encoding='utf-8', mode='w') as file:
        file.write(str(traceback.format_exc())+"\n\n"+str(cs)+"\n\n"+str(queues)+"\n\n"+str(players))
    await user.send("快过来修我！")
    await user.send("错误文件：", file=discord.File(efn, "err.txt"))

    await ctx.send(replacetrans("developer_notified",ctx.author.id))

@tasks.loop(seconds=11451 if not key["devmode"] else 15  )
async def writeplays():
    await atri.change_presence(activity=discord.Activity(type=discord.ActivityType.listening,name="主人的命令||" + name[0] + "play <歌曲>||支持网易云，哔哩哔哩，youtube，ニコニコ"))
    await asyncio.sleep(5)
    print("主人我正在保存数据喵~")
    with codecs.open(dirpath + "./plays.json", encoding='utf-8', mode='w') as f:
        f.write(json.dumps(plays))
    with codecs.open(dirpath + "./haogan.json", encoding='utf-8', mode='w') as f:
        f.write(json.dumps(userdata))
    with codecs.open(dirpath + "./waifulist.txt", encoding='utf-8', mode='w') as f:
        f.write(str(waifulist))
    with codecs.open(dirpath + "blacklist.json", encoding="utf-8", mode="w") as f:
        f.write(json.dumps(blacklist))
    with codecs.open(dirpath + "langpref.json", encoding="utf-8", mode="w") as f:
        f.write(json.dumps(langpref))
    print("数据保存完毕，大家的信息我已经完美的记录下来的喵~")
    # print("主人我正在打扫房间~")
    # # for f in os.listdir(dirpath+"ytdltemp"):
    # #     os.remove(os.path.join(dirpath+"ytdltemp", f))
    # # for filename in os.listdir(dirpath+"songcache"):
    # #     if os.stat(filename).st_size==0:
    # #         os.remove(filename)
    # #1
    # print("主人，房间已经清扫的干干净净了喵~")

def startatri():

    atri.run(key["key"])

if __name__ == '__main__':
    thread = threading.Thread(target=serve,args=[app],kwargs={ "host":"0.0.0.0", "port":key["serverport"]})
    thread.start()
    startatri()
    thread.join()
