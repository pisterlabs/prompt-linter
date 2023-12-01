import PyPDF2

from langchain.chat_models import ChatOpenAI
import langchain
import bs4
import requests
import io
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import tiktoken
import aiohttp
from reliablegpt import reliableGPT
import openai
from gptcache import cache
from gptcache.adapter import openai
from dotenv import dotenv_values


cache.init()
cache.set_openai_key()
with open('email.txt','r') as f:
    email =  f.readline()
openai.ChatCompletion.create = reliableGPT(openai.ChatCompletion.create, user_email= email)

executor = ThreadPoolExecutor(max_workers=5)

content_string = '''\
You are SummarizeGPT, a LLM that summarizes research papers into their main ideas.
Suggest a title for the summarised content and write a concise and comprehensive summary of the paper.

The format should be Title:title content\n\nSummary:summary content
 '''

tagger_string = '''\
You are TaggerGPT, a LLM that tags research papers with their respective fields. Given the content of a paper, suggest a title and tag the paper with the appropriate field.
The given tags allowed are the following: Language, Vision, RL, Alignment, Robotics, Audio and Miscellaneous.
You can give more than 1 tag to the paper if you think it is appropriate. You will only respond with the tag, and will delimit multiple tags with commas. 

Example output: Tags: Vision, Language
'''

def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = tokenizer.encode(text)
    return len(tokens)

tag_prompt_len = count_tokens(tagger_string)

template_len = count_tokens(content_string)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3200 - template_len,
    chunk_overlap=100,
    length_function=count_tokens,
    separators=["\n\n","\n",'']
)



def scrape_hf(url,current_posts= []):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    sections = soup.find_all('article', class_ = 'flex flex-col overflow-hidden rounded-xl border')
    hyperlink_texts = [section.find('h3') for section in sections]
    upvotes = [section.find('div', class_ = 'leading-none').text for section in sections]
    upvotes = list(map(lambda x: int(x) if x !='-' else 0,upvotes))
    results = []
    filtered = []
    for count,text in enumerate(hyperlink_texts):
        data = text.find('a', class_ = 'cursor-pointer')
        results.append((upvotes[count],data.text,data['href'].replace('papers','pdf')))
    print(current_posts)
    base_filtered = list(filter(lambda x: x[1] not in current_posts,results))
    print('filtr\n',base_filtered)
    filtered = list(filter(lambda x: x[0] >8,base_filtered))
    return filtered if len(filtered)>=3 else base_filtered[:3]

async def scrape_arxiv(codes,batch_size = 5):
    texts = []
    async with aiohttp.ClientSession() as session:
        for count in range(0,len(codes),batch_size): #Loads 5 papers worth of text, and then sends them to the summarizer
            links =  codes[count:count+batch_size] 
            tasks = [process_pdf(session, 'https://arxiv.org'+link+'.pdf') for link in links]
            batch_results = await asyncio.gather(*tasks)
            texts.extend(batch_results)
    return texts
        
           
async def download_pdf(session,url):
    try:
        async with session.get(url) as response:
            return await response.read()
    except Exception as e:
        return e
    

async def parse_pdf(session,url):
    pdf_data = await download_pdf(session,url)
    if isinstance(pdf_data, Exception):
        return pdf_data

    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(pool, parse_pdf_sync, pdf_data)
    return result

def parse_pdf_sync(pdf_data):
    file = io.BytesIO(pdf_data)
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text+=page.extract_text()
    return text


async def process_pdf(session,url):
    pdf_text = await  parse_pdf(session,url)
    if isinstance(pdf_text, Exception):
        return pdf_text
    
    result = await summarize(pdf_text)
    tags = await tagger(result)
    return (tags,result)

##Count tokens then summarize
async def summarize(text):
    portions = text_splitter.split_text(text)
    docs = []
    for portion in portions:
        if 'References\n' in portion:
            portion = portion.split('References\n')[0]
            docs.append(portion)
            break
        else:
            docs.append(portion)
    results = [async_generate(content_string,doc) for doc in docs]
    results = await asyncio.gather(*results)
    out = ''.join(results) if len(docs) == 1 else '\n\n'.join(results)
    if count_tokens(out) > 4000:
        print('Summarizing summary......')
        return await summarize(out)
    else:
        results = await async_generate(content_string,out)
        print('Paper summarized!')
    return results

async def async_generate(prompt,text):
    print('Calling API to generate.......')
    messages = [{'role':'system','content':prompt},{'role':'user','content':text}]
    res = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = messages,
    )
    return res['choices'][0]['message']['content']

async def tagger(text):
    while count_tokens(text) >= 4000:
        text = text[:-50] #Truncate the text until it is below 4000 tokens
    results = await async_generate(tagger_string,text)
    return results

##### DISCORD LOGIC ########

def format_post(post):
    if '\n\nSummary:' in post:
        post = 'Summary:'+post.split('\n\nSummary:')[1]
        return post
    elif '\n\n' in post:
        post = post.split('\n\n')[1]
        return post

def format_tags(tags):
    approved_tags = {'Language','Vision','RL','Alignment','Robotics','Audio','Miscellaneous'}
    print(tags)
    ##Validate and format tags
    valid_tags = set()
    if 'Tags:' in tags:
        post_tags = tags.split('Tags:')[1].split(',')
        valid_tags = {tag.strip() for tag in post_tags} & approved_tags
    else:
        post_tags = tags.split(',')
        valid_tags = {tag.strip() for tag in post_tags} & approved_tags
    return valid_tags if valid_tags else 'Miscellaneous'
        
async def main(post_history = []):
    res = scrape_hf('https://huggingface.co/papers',post_history)
    if res == []:
        return
    _,names,codes = zip(*res)
    tags,posts = zip(*await scrape_arxiv(codes))
    posts = list(map(format_post,posts))
    tags = list(map(format_tags,tags))
    print(tags)
    results = []
    for post in zip(names,codes,posts,tags):
        results.append(post)
    return results
        
#### DISCORD LOGIC #####
import discord
from discord.ext import commands
from datetime import datetime, timedelta,time

intents = discord.Intents.all()
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

class MyHelp(commands.HelpCommand):
    async def send_bot_help(self, mapping):
        embed = discord.Embed(title="Help")
        for cog, commands in mapping.items():
           command_signatures = [self.get_command_signature(c) for c in commands]
           if command_signatures:
                cog_name = getattr(cog, "qualified_name", "No Category")
                embed.add_field(name=cog_name, value="\n".join(command_signatures), inline=False)

        channel = self.get_destination()
        await channel.send(embed=embed)


    async def send_command_help(self, command):
        embed = discord.Embed(title=self.get_command_signature(command), color=discord.Color.random())
        if command.help:
            embed.description = command.help
        if alias := command.aliases:
            embed.add_field(name="Aliases", value=", ".join(alias), inline=False)

        channel = self.get_destination()
        await channel.send(embed=embed)

bot.help_command = MyHelp()

@bot.event
async def on_ready():
    print('We have logged in as {0.user}'.format(bot))


async def output_paper(ctx,res,code):
    if isinstance(res, Exception):
            await ctx.send('Invalid link. Please try again.')
            return
        
    tags,posts = zip(*res)
    posts = list(map(format_post,posts))
    tags = list(map(format_tags,tags))
    if not isinstance(code,list):
        code = [code]
    if len(tags) >1:
        tags = ','.join(tags)
    else:
        tags = tags[0]
    for post in zip(code,posts,tags):
        await ctx.send(
            f"{post[0]}:\n\n{post[1]}\nTags given: {post[2]}\n"
        )
        

@bot.command(help = 'Summarizes a paper given the link to the paper')
async def summarize_paper(ctx,code):
    if 'https://arxiv.org/pdf/' in code:
        code = code.split('https://arxiv.org/pdf/')[-1]
        code = code.split('.pdf')[0]
        res = await scrape_arxiv(['/pdf/'+code])
        print(res)
        await output_paper(ctx,res,code)
    elif 'https://arxiv.org/abs/' in code:
        code = code.split('https://arxiv.org/abs/')[-1]
        res = await scrape_arxiv(['/pdf/'+code])
        await output_paper(ctx,res,code)
    elif code.split('.')[0].isdigit() and code.split('.')[1].isdigit():
        res = await scrape_arxiv(['/pdf/'+code])
        await output_paper(ctx,res,code)
    else:
        await ctx.send('Invalid link. Please try again.')

async def daily_scrape():
    while True:
        now = datetime.now()
        night_time = datetime.combine(now + timedelta(days=1),time(18,0,0) )
        await asyncio.sleep((night_time-now).seconds)
        channel = bot.get_channel(1123830409977938034)
        channel_tags = channel.available_tags
        threads = channel.threads
        threads = list(map(lambda x: x.name,threads))
        

        results = await main(threads)
        if results == None:
            print('No papers for today!')
            return
        for result in results:
            name,code,post,tags = result
            tag_dict = {tag.name: tag for tag in channel_tags}
            output_tags = []
            for tag in tags:
                output_tags.append(tag_dict[tag])


            thread = await channel.create_thread(name = name,
                                                content = f"https://arxiv.org{code}.pdf\n\n{post[:3900]}",
                                                applied_tags = output_tags)
            for n in range(3900,len(post),3900):
                await thread.send(
                f"{post[n:(n+3900)]}"
            )


async def find_tags():
    channel = bot.get_channel(1123830409977938034)
    channel_tags = channel.available_tags
    return channel_tags

@bot.command(help = 'Find all threads in given channel')
async def find_threads(ctx):
    channel = bot.get_channel(1123830409977938034)
    threads = channel.threads
    for thread in threads:
        print(thread.name)

if __name__ == '__main__':
    config = dotenv_values('.env')
    token = config['BOT_TOKEN']
    bot.loop.create_task(daily_scrape())
    bot.run(token)
