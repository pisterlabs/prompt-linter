import interactions
import openai
import os
import httpx
import sys
import random
import json

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
openai.api_key = "" # 非公开内容
TOKEN = "" # 非公开内容
st_l_file = "st_l.json"
if not os.path.exists(st_l_file):
    json.dump({"normal":{},"r18":{},"all":{}},open(st_l_file,"w"))

bot = interactions.Client(token=TOKEN)
bot.http.proxy = ("http://127.0.0.1:7890",None) # type: ignore

async def getImage(r18,tag=""):
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.lolicon.app/setu/v2?r18="+r18+tag)
        d = r.json()["data"][0]
        if d["title"] == "" and d["tag"] == []:
            return await getImage(r18,tag)
        img = await client.get(d["urls"]["original"].replace("i.pixiv.re","i.pixiv.cat"))
        cachefile_fn = f"cachefile{random.randint(10000,99999)}.{d['ext']}"
        cachefile = open(cachefile_fn,"wb")
        cachefile.write(img.content)
        cachefile.close()
        # os.remove(cachefile_fn)
        return [f'PID: {d["pid"]}\nTITLE: {d["title"]}',cachefile_fn]
async def getUser(uid):
    try:
        return bot.get_user(uid).display_name # type: ignore
    except:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"https://discord.com/api/users/{uid}",headers={
                "User-Agent":"DiscordBot",
                "Authorization":f"Bot {TOKEN}"
            })
        return r.json()["username"]
def getSt_l():
    return json.load(open(st_l_file,"r"))
def addSt_l(uid,r18):
    uid = str(uid)
    st_l = getSt_l()
    if r18 == "1":
        _type = "r18"
    else:
        _type = "normal"
    try:
        st_l[_type][uid] += 1
        st_l["all"][uid] += 1
    except:
        st_l[_type][uid] = 1
        st_l["all"][uid] = 1
    json.dump(st_l,open(st_l_file,"w"))

@interactions.slash_command(
    name="cg",
    description="Ask ChatGPT",
)
@interactions.slash_option(
    name="content",
    description="The text you want to ask ChatGPT",
    required=True,
    opt_type=interactions.OptionType.STRING
)
async def cg(ctx: interactions.ComponentContext, content: str):
    await ctx.defer()
    message = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [{"role":"user","content":content}],
        )["choices"][0]["message"]["content"] # type: ignore
    await ctx.send(message)

@interactions.slash_command(
    name="st-r",
    description="Random pixiv image",
)
@interactions.slash_option(
    name="tag",
    description="Image tag",
    required=False,
    opt_type=interactions.OptionType.STRING
)
async def st_r(ctx: interactions.ComponentContext, tag= ""):
    await ctx.defer()
    tag = "|".join(tag.strip().split(" "))
    if tag != "":
        tag = "&tag="+tag
    image = await getImage("0",tag)
    addSt_l(ctx.author.id.numerator,"0")
    await ctx.send(image[0],file=image[1])
    os.remove(image[1])

@interactions.slash_command(
    name="st-r-r18",
    description="Random pixiv (nsfw)",
    nsfw=True
)
@interactions.slash_option(
    name="tag",
    description="Image tag",
    required=False,
    opt_type=interactions.OptionType.STRING
)
async def st_r_r18(ctx: interactions.ComponentContext, tag= ""):
    await ctx.defer()
    tag = "|".join(tag.strip().split(" "))
    if tag != "":
        tag = "&tag="+tag
    image = await getImage("1",tag)
    addSt_l(ctx.author.id.numerator,"1")
    await ctx.send(image[0],file=image[1])
    os.remove(image[1])

@interactions.slash_command(
    name="st-l",
    description="Display the times user use /st-r and /st-r-r18"
)
@interactions.slash_option(
    name="_type",
    description="normal | r18 | all (default: all)",
    required=False,
    opt_type=interactions.OptionType.STRING,
    choices=[
        interactions.SlashCommandChoice(name="all", value="all"),
        interactions.SlashCommandChoice(name="normal", value="normal"),
        interactions.SlashCommandChoice(name="r18", value="r18")
    ]
)
async def st_l_cmd(ctx: interactions.ComponentContext, _type="all"):
    await ctx.defer()
    st_l = getSt_l()
    st_l = st_l[_type]
    st_l = sorted(st_l.items(),key=lambda x:x[1],reverse=True)
    msg = f"### st-r 次数排行榜 ({_type}):\n"
    uid = str(ctx.author.id.numerator)
    you_msg = f"∞. **{ctx.author.display_name}:** `0`" # type: ignore
    for i in range(len(st_l)):
        msg += (cache := f"{i+1}. **{await getUser(st_l[i][0])}:** `{st_l[i][1]}`\n")
        if st_l[i][0] == uid:
            you_msg = cache.replace("\n","")
    msg += "------------------------------\n"
    msg += you_msg
    await ctx.send(msg)

@interactions.slash_command(
    name="test",
    description="Get your user id (test)",
)
async def test(ctx: interactions.ComponentContext):
    await ctx.defer()
    await ctx.send(f"Your ID = {ctx.author.id.numerator}\nYour Display Name = {await getUser(ctx.author.id.numerator)}") # type: ignore

@interactions.slash_command(
    name="echo",
    description="Echo some text"
)
@interactions.slash_option(
    name="content",
    description="The text you want to echo",
    required=True,
    opt_type=interactions.OptionType.STRING
)
async def echo(ctx: interactions.ComponentContext, content: str):
    await ctx.send(content) # type: ignore

def restart_program():
  python = sys.executable
  os.execl(python, python, *sys.argv)

bot.start()
restart_program()