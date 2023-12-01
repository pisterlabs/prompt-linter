import argparse
from argparse import ArgumentParser
from itertools import chain
import torch.nn.functional as F
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from discord_webhook import DiscordWebhook, DiscordEmbed
import pickle, discord, re, random, os, dbl, warnings, torch
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import time, requests, datetime
from datetime import date
from discord.ext.tasks import loop
import json, build_versions

global client, config
client = discord.Client()
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
with open('config.json') as json_file:
    config = json.load(json_file)
    
def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, int(top_k))[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())
    return current_output

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def similarity(X,Y):
    X_list = word_tokenize(X)  
    Y_list = word_tokenize(Y) 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 

    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 

    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0

    for i in range(len(rvector)): 
            c+= l1[i]*l2[i]
    try:
        cosine = c / float((sum(l1)*sum(l2))**0.5) 
    except:
        cosine = 0
    return cosine

def avg_similarity(max_history,h):
    total = 0
    h=h[:-1]
    if len(h) < max_history:
        h=h[1:]
    for i,val in enumerate(h):
        if i % 2 == 0:
            try:
                total += similarity(h[i],h[i+2])
            except:
                pass
    total=total/(len(h)//2)
    return total

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
parser.add_argument("--model_checkpoint", type=str, default="run0", help="Path, url or short name of the model")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

parser.add_argument("--temperature", type=int, default=0.85, help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=40, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")

parser.add_argument("--seed", type=int, default=random.randint(0,9999999999), help="Seed")
parser.add_argument("--auto-seed", type=str2bool, default=True, help="auto-seeding")
parser.add_argument("--max_history", type=int, default=4, help="Number of previous utterances to keep in history")
parser.add_argument("--no_sample", type=str2bool, default=False, help="Set to use greedy decoding instead of sampling")
parser.add_argument("--max_length", type=int, default=10, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
args = parser.parse_args()
args1 = parser.parse_args()

loaded=0
try:
    if loaded != 1:
        if args.model_checkpoint == "":
            raise ValueError("Interacting requires passing a finetuned model_checkpoint")
        
        if args.seed != 0:
            random.seed(args.seed)
            torch.random.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

        print("Get pretrained model and tokenizer")
        tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
        tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
        model = model_class.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

        personalities = pickle.load(open(os.path.join(args.model_checkpoint, "versions.p"), "rb"))
        personality = random.choice(personalities)
        print("Selected personality:", tokenizer.decode(chain(*personality)))

        loaded=1
        del loaded
except Exception as e:
    print(e)
    pass
def get_history(message):
    try:
        hist=""
        history= pickle.load(open("hist/"+str(message.guild.id)+".p", "rb"))["history"]
        for i, p in enumerate(history):
            if i % 2 == 0:
                hist+="> "+tokenizer.decode(p, skip_special_tokens=True)+"\n"
            else:
                hist+=tokenizer.decode(p, skip_special_tokens=True)+"\n"
        if len(hist) == 0:
            return "No History!"
        else:
            return hist
    except Exception as e:
        print(e)
        return "No History!"

global dbli
dbli=dbl.DBLClient(client, config["dbltoken"])

@loop(seconds=1800)
async def update_guilds():
    global dbli, client
    print("Posting a guild count of " + str(len(client.guilds)))
    await dbli.post_guild_count()
    requests.post("https://bots.ondiscord.xyz/bot-api/bots/410253782828449802/guilds", data = 'Authorization: '+config["botsondiscord"]+'\nContent-Type: application/json\n{"guildCount": '+str(len(client.guilds))+'}')
    requests.post("https://discord.bots.gg/api/v1/bots/410253782828449802/stats", data = 'Authorization: '+config["dbots.gg"]+'\nContent-Type: application/json\n{"guildCount": '+str(len(client.guilds))+'}')

@client.event
async def on_guild_join(guild):
    try:
        webhook = DiscordWebhook(url=config["logchannel"], avatar_url=str(guild.icon_url), username=str(guild.name))
        embed = DiscordEmbed(title="Joined guild", description=str(guild.id), color=0xaaff88)
        embed.set_author(name=str(guild),icon_url=str(guild.icon_url))
        embed.set_footer(text=str(time.strftime('%X %x %Z')))
        webhook.add_embed(embed)
        webhook.execute()
    except:
        pass

@client.event
async def on_guild_remove(guild):
    try:
        os.remove("hist/"+str(guild.id)+".p")
    except:
        pass
    try:
        webhook = DiscordWebhook(url=config["logchannel"], avatar_url=str(guild.icon_url), username=str(guild.name))
        embed = DiscordEmbed(title="Left guild", description=str(guild.id), color=0xff9988)
        embed.set_author(name=str(guild),icon_url=str(guild.icon_url))
        embed.set_footer(text=str(time.strftime('%X %x %Z')))
        webhook.add_embed(embed)
        webhook.execute()
    except:
        pass

@client.event
async def on_ready():
    global personality, tokenizer, model, dbli
    print('Logged in as '+client.user.name+' (ID:'+str(client.user.id)+') | Connected to '+str(len(client.guilds))+' servers | Connected to '+ str(len(set(client.get_all_members()))) +' users')
    print('--------')
    print("Discord.py verison: " + discord.__version__)
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="everyone talk ✨✨"))
    update_guilds.start()

prefix=config["prefix"]
current_version=1
global t1, settings, history,user_version
try:
    udata=pickle.load(open("hist/user/users.p", "rb"))
except:
    pickle.dump({"message_total":{},"message_rate":{},"users":{}}, open("hist/user/users.p", "wb"))

@client.event
async def on_message(message):
    if message.guild==None and message.author.bot == False:
        embed=discord.Embed(title="DM integration", url="https://www.notion.so/jadeai/1c0f1d42eb6345b58013a1be35e47793?v=d45f7f3d26e140c995f8a9021564bb99", description="Dms are not supported yet! when they are, they will require a upvote on top.gg and a confimed server referral to a server with 10+ non-bot members", color=0x80ff80)
        await message.channel.send(embed=embed)
    elif message.author.bot == False:
        global personality, tokenizer, model, client, t1, settings, history,user_version
        if message.content.lower().startswith(prefix):
            history=[]
            settings=args1
            user_version=-1
            try:
                user_status=await dbli.get_user_vote(user_id=message.author.id)
            except:
                user_status=False
            udata=pickle.load(open("hist/user/users.p", "rb"))
            try:
                user_data=udata["users"][message.author.id]
            except:
                user_data={"timestamp": time.time()-30, "message_count":0}
            try:
                for key, value in pickle.load(open("hist/"+str(message.guild.id)+".p", "rb")).items():
                    globals()[str(key)]=value
            except Exception as e:
                args.seed=random.randint(0,9999999999)
                t1=time.time()-30
                pickle.dump({"t1":t1,"settings":args,"history":[], "user_version":current_version}, open("hist/"+str(message.guild.id)+".p", "wb"))
            if user_version != current_version:
                for version_num in range(user_version+1, current_version+1):
                    try:
                        await message.channel.send(embed=build_versions.version_message(version_num, client, prefix))
                    except: pass
        if message.content.lower() == prefix+"-h":
            embed=await build_versions.make_help(dbli, client, prefix)
            await message.channel.send(embed=embed, delete_after=150)
            try:
                await message.delete()
            except:
                pass
        elif message.content.lower() == prefix+"-p":
            embed=discord.Embed(title="User Profile: "+ str(message.author), url="https://jadeai.ml", color=0x80ff80)
            embed.set_thumbnail(url=message.author.avatar_url)
            embed.add_field(name="Last seen", value= str(datetime.datetime.fromtimestamp(user_data["timestamp"]).strftime('%X %x')) + time.strftime(" %Z"), inline=False)
            embed.add_field(name="Number of Messages", value= str(user_data["message_count"]), inline=False)
            embed.set_footer(text="Global Total: " + str(udata["message_total"][str(date.today())]))
            await message.channel.send(embed=embed, delete_after=150)
            try:
                await message.delete()
            except:
                pass
        elif message.content.lower() == prefix+"-v":
            embed=discord.Embed(title="Voting Link", url="https://top.gg/bot/410253782828449802/vote", color=0x80ff80)
            embed.set_image(url=await dbli.get_widget_large(client.user.id))
            if await dbli.get_user_vote(user_id=message.author.id):
                embed.set_footer(text="Thanks for supporting Jade!")
            else:
                embed.set_footer(text="You have yet to vote for Jade!")
            embed.set_author(name=str(message.author), icon_url=message.author.avatar_url)
            await message.channel.send(embed=embed, delete_after=100)
            try:
                await message.delete()
            except:
                pass
        elif message.content.lower() == prefix+"-s":
            history=get_history(message)
            settings=vars(settings)
            embed= discord.Embed(title="Settings", url="https://jadeai.ml", description="__                                                                                      __\nServer-Side Settings  🔒", color=0x80ff80)
            embed.add_field(name="model", value=str(settings["model"]), inline=True)
            embed.add_field(name="model_checkpoint", value=str(settings["model_checkpoint"]), inline=True)
            embed.add_field(name="device", value=str(settings["device"]), inline=True)
            embed.add_field(name="__                                                                                                                                   __", value="User-Changable Settings  🔓", inline=False)
            embed.add_field(name="temperature", value=str(settings["temperature"])+"/1", inline=True)
            embed.add_field(name="top_k", value=str(settings["top_k"]), inline=True)
            embed.add_field(name="top_p", value=str(settings["top_p"])+"/1", inline=True)
            if user_status:
                val="Supporter-Only Settings  🔓"
            else:
                val="Supporter-Only Settings  🔐 [vote for her here](https://top.gg/bot/410253782828449802/vote)"
            embed.add_field(name="__                                                                                                                                   __", value=val, inline=False)
            embed.add_field(name="seed", value=str(settings["seed"]), inline=True)
            embed.add_field(name="auto_seed", value=str(settings["auto_seed"]), inline=True)
            embed.add_field(name="max_history", value=str(settings["max_history"])+'/10', inline=True)
            embed.add_field(name="max_length", value=str(settings["max_length"])+"/20", inline=True)
            embed.add_field(name="no_sample", value=str(settings["no_sample"]), inline=True)
            embed.add_field(name="​", value="​", inline=True)
            embed.add_field(name="__                                                                                                                                   __", value="History", inline=False)
            if len(get_history(message).replace("> ","").split("\n")) >=4:
                embed.add_field(name="Jade similarity score: `"+ str(avg_similarity(settings["max_history"],get_history(message).replace("> ","").split("\n")))+"`", value=get_history(message), inline=False)
            else:
                embed.add_field(name="Jade similarity score: `NAN`", value=get_history(message), inline=False)
            await message.channel.send(embed=embed, delete_after=300)
            try:
                await message.delete()
            except:
                pass
        elif message.content.lower().startswith(prefix+"-s "):
            parameter=message.content.lower()[len(prefix)+3:].split(" ")
            any_changes=False
            if len(parameter) == 2:
                alt_settings=vars(settings)
                server=["model", "model_checkpoint", "device"]
                client_side=["temperature","top_k","top_p"]
                privledged=["no_sample","seed", "auto_seed", "max_history", "max_length"]
                limiters={"temperature":{"max": 1, "type":float}, "top_k":{"max": float("inf"), "type":int}, "top_p":{"max": 1, "type":float},
                "no_sample":{"type":str2bool}, "seed":{"max": float("inf"), "type":int}, "auto_seed":{"type":str2bool}, 
                "max_history":{"max": 10, "type":int}, "max_length":{"max": 20, "type":int}}
                if parameter[0] in server:
                    embed=discord.Embed(title="Settings", description="`"+str(parameter[0])+"` is a server-side setting, and cannot be changed.", color=0x80ff80)
                elif parameter[0] in privledged and user_status==False:
                    embed=discord.Embed(title="Settings", description="`"+str(parameter[0])+"` is a supporter-only setting. [vote for Jade on top.gg](https://top.gg/bot/410253782828449802/vote)", color=0x80ff80)
                elif (parameter[0] in client_side) or parameter[0] in privledged and user_status==True:
                    ch=limiters[parameter[0]]["type"](parameter[1])
                    if limiters[parameter[0]]["type"] == float or limiters[parameter[0]]["type"] == int:
                        if limiters[parameter[0]]["max"] >= ch and ch >= 0:
                            embed=discord.Embed(title="Settings", description="`"+str(parameter[0])+"` changed from `"+str(alt_settings[parameter[0]])+"` to `"+str(parameter[1])+"`", color=0x80ff80)
                            embed.set_footer(text="Default setting: "+str(vars(args1)[parameter[0]]))
                            alt_settings[parameter[0]]=ch
                            any_changes=True
                        else:
                            embed=discord.Embed(title="Settings", description="`"+str(parameter[0])+"` could not be changed from `"+str(alt_settings[parameter[0]])+"` to `"+str(parameter[1])+"` becasue it is `<= 0` or `>= "+str(limiters[parameter[0]]["max"])+"`", color=0x80ff80)
                            embed.set_footer(text="Default setting: "+str(vars(args1)[parameter[0]]))
                    else:
                        embed=discord.Embed(title="Settings", description="`"+str(parameter[0])+"` changed from `"+str(alt_settings[parameter[0]])+"` to `"+str(ch)+"`", color=0x80ff80)
                        embed.set_footer(text="Default setting: "+str(vars(args1)[parameter[0]]))
                        alt_settings[parameter[0]]=ch
                        any_changes=True
                else:
                    embed=discord.Embed(title="Settings", description="`"+str(parameter[0])+"` is not a valid setting.", color=0x80ff80)
                pickle.dump({"t1":t1, "settings":settings,"history":history, "user_version":user_version}, open("hist/"+str(message.guild.id)+".p", "wb"))
            else:
                embed=discord.Embed(title="Settings", description="`"+str(parameter)+"` contains more than two parts.", color=0x80ff80)
            await message.channel.send(embed=embed, delete_after=150)
            try:
                await message.delete()
            except:
                pass
            if any_changes:
                try:
                    settings=vars(settings)
                    webhook = DiscordWebhook(url=config["logchannel"], avatar_url=str(message.guild.icon_url), username=str(message.guild.name))
                    embed= DiscordEmbed(title="Settings", description="__                                                                                      __", color=0x80ff80)
                    embed.add_embed_field(name="model", value=str(settings["model"]))
                    embed.add_embed_field(name="model_checkpoint", value=str(settings["model_checkpoint"]))
                    embed.add_embed_field(name="device", value=str(settings["device"]))
                    embed.add_embed_field(name="temperature", value=str(settings["temperature"])+"/1")
                    embed.add_embed_field(name="top_k", value=str(settings["top_k"]))
                    embed.add_embed_field(name="top_p", value=str(settings["top_p"])+"/1")
                    embed.add_embed_field(name="seed", value=str(settings["seed"]))
                    embed.add_embed_field(name="auto_seed", value=str(settings["auto_seed"]))
                    embed.add_embed_field(name="max_history", value=str(settings["max_history"])+'/10')
                    embed.add_embed_field(name="max_length", value=str(settings["max_length"])+"/20")
                    embed.add_embed_field(name="no_sample", value=str(settings["no_sample"]))
                    embed.add_embed_field(name="​", value="​")
                    embed.add_embed_field(name="__                                                                                      __\n", value=time.strftime('%X %x %Z'))
                    webhook.add_embed(embed)
                    webhook.execute()
                except:
                    pass
        elif message.content.lower().startswith(prefix+"-r"):
            parameter=message.content.lower()[len(prefix)+3:]
            desc="Reset settings and history"
            h=["-h","hist","history"]
            s=["-s","settings"]
            for s1 in s:
                if s1 in parameter:
                    desc="Reset settings"
                    settings=args1
            for h1 in h:
                if h1 in parameter:
                    desc="Reset history"
                    history=[]
            if parameter=="":
                history=[]
                settings=args1
            pickle.dump({"t1":t1, "settings":settings,"history":history, "user_version":user_version}, open("hist/"+str(message.guild.id)+".p", "wb"))
            embed=discord.Embed(title="Reset", description=desc, color=0x80ff80)
            await message.channel.send(embed=embed, delete_after=100)
            try:
                await message.delete()
            except:
                pass
            try:
                webhook = DiscordWebhook(url=config["logchannel"], avatar_url=str(message.guild.icon_url), username=str(message.guild.name))
                embed = DiscordEmbed(title="Server settings reset", description=time.strftime('%X %x %Z'), color=0x80ff80)
                embed.set_author(name=str(message.author), icon_url=str(message.author.avatar_url))
                webhook.add_embed(embed)
                webhook.execute()
            except:
                pass
        elif message.content.lower().startswith(prefix):
            if user_status:
                ratelimit=2
            else:
                ratelimit=8
            if round(time.time())-user_data["timestamp"] > ratelimit:
                await message.channel.trigger_typing()
                raw_text = message.content[len(prefix):][:100].lower().strip()
                raw_text = re.sub(r"([?.!,])", r" \1 ", raw_text)
                raw_text = re.sub(r'[" "]+', " ", raw_text)
                raw_text = re.sub(r"[^a-zA-Z0-9?.!,\'%\s\/#]+", "", raw_text)
                raw_text = re.sub(r"(\s+){2,}", " ", raw_text)
                raw_text = raw_text.strip()
                history.append(tokenizer.encode(raw_text.replace("\n"," ")))
                with torch.no_grad():
                    out_ids = sample_sequence(personality, history, tokenizer, model, settings)
                history.append(out_ids)
                history = history[-(2*args.max_history+1):]
                if len(get_history(message).replace("> ","").split("\n")) >=4:
                    if avg_similarity(settings.max_history,get_history(message).replace("> ","").split("\n")) >= 0.3 and settings.auto_seed == True:
                        settings.seed=random.randint(0,9999999999)
                pickle.dump({"t1":round(time.time()),"settings":settings,"history":history,"user_version":current_version}, open("hist/"+str(message.guild.id)+".p", "wb"))
                user_data["message_count"]+=1
                user_data["timestamp"]=round(time.time())
                udata["users"][message.author.id]=user_data
                new_data=udata["message_rate"]
                new_total=udata["message_total"]
                try:
                    new_data[str(date.today())]=udata["message_rate"][str(date.today())]+1
                    new_total[str(date.today())]=udata["message_total"][str(date.today())]+1
                except:    
                    new_data[str(date.today())]=1
                    new_total[str(date.today())]=udata["message_total"][str(date.today()-datetime.timedelta(days = 1))]+1
                pickle.dump({"message_total":new_total,"message_rate":new_data,"users":udata["users"]}, open("hist/user/users.p", "wb"))
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                await message.channel.send(out_text)
                try:
                    webhook = DiscordWebhook(url=config["logchannel"], avatar_url=str(message.guild.icon_url), username=str(message.guild.name))
                    embed = DiscordEmbed(title="__                                                                                                         __\n\n"+str(message.author)+": "+raw_text.replace("\n"," "), description="Jade: "+out_text, color=0x80ff80)
                    embed.add_embed_field(name="__                                                                                                                        __", value=time.strftime('%X %x %Z'))
                    embed.set_author(name=str(message.author), icon_url=str(message.author.avatar_url))
                    webhook.add_embed(embed)
                    webhook.execute()
                except:
                    pass
            else:
                embed = discord.Embed(title="Ratelimit", description="Calm down or [upvote Jade](https://top.gg/bot/410253782828449802/vote) before trying again!", color=0x80ff80)
                embed.set_footer(text="Try again in "+str(round(time.time()-t1))+" seconds")
                embed.set_author(name=str(message.author), icon_url=str(message.author.avatar_url))
                await message.channel.send(embed=embed, delete_after=50)
                try:
                    await message.delete()
                except:
                    pass

client.run(config["token"])