import asyncio
from bs4 import BeautifulSoup
from collections import OrderedDict
from csv import writer
import html
import json
import openai
import os
from pyppeteer import launch
from pyppeteer.errors import PyppeteerError
from pyppeteer_stealth import stealth
import random
import re
import requests
from termcolor import cprint
import time
import tldextract


# download the feed using pyppeteer (some of the sites will return junk if you just do a standard download)
def get_feed(url): 
    async def get_feed_raw(): #startin up the pyppeteer
        browser = await launch()
        page = await browser.newPage()
        await stealth(page) #lookin at you indeed
        await page.goto(url)
        feed_raw = await page.content()
        await browser.close()
        return feed_raw

    try:
        feed_raw = asyncio.get_event_loop().run_until_complete(get_feed_raw()) #get the feed output
    except:
        return False

    feed_raw = html.unescape(feed_raw) #unescape the feed, sometimes feeds come from pyppeteer with the xml brackets encoded

    return feed_raw


#collect the links from the rss feed
def get_links_from_feed(feed_raw): 
    raw_links = re.findall('<link>h[^<]*</link>', feed_raw) #using regex for the feed link collection. feeds are that come out of pyppeteer are malformed since chrome formats them

    feed_domain_raw = re.sub('<[^<]+?>', '',raw_links[0])
    feed_domain = get_domain (feed_domain_raw)

    output_links = []
    output_links_clean = []
    if len(raw_links)>1: #make sure we actually got some links
        for link in raw_links:
            link = re.sub('</?link>','', link) #remove the link tag from around the link
            link = html.unescape(link) #xml requires that ampersands be escaped, fixing that for our links
            output_links.append(link) #add the clean link to our output 

        output_links.pop(0) #first link in an rss feed is a link to the site associated with the feed https://www.rssboard.org/rss-draft-1#element-channel-link

        if output_links[0] == "https://www.indeed.com/": #indeed sometimes outputs an extra link to itself at the start of the file
           output_links.pop(0) 

        #going to clean the links a bit, most links have some query string stuff that isn't needed, exceept indeed, which has the job id in the query string.
        #doing this so that we lose all the unique tracking IDs or whatever on the link, so later on it'll be easier to see if we've already looked at a link.
        for output_link in output_links:
            domain = get_domain(output_link) #get domain by itself
            if domain =="indeed":
                output_links_clean.append(output_link.split("&rtk=", 1)[0]) #indeed keeps the job id in the query string, but the text after 'rtk' changes (and isn't needed), which would break our system of ignoring previously processed links
            else:
                output_links_clean.append(output_link.split("?", 1)[0]) #toss everything after the question mark
    else:
        print(f"      Found    {len(output_links_clean)} jobs on {feed_domain.capitalize()}")
        return 0 #if we didn't get any links, we'll return zero
    
    feed_domain = get_domain (feed_domain_raw)

    print(f"      Found    {len(output_links_clean)} jobs on {feed_domain.capitalize()}")
    return output_links_clean

#download the feed using pyppeteer (some of the sites will return junk if you just do a standard download)
def get_jorb(url):

    #this is a list of the html classes that wrap the job description on the various sites
    #find the smallest div with a class that wraps the entire job description
    jobsite_container_class={ 
        "indeed": "jobsearch-JobComponent",
        "chronicle": "mds-surface",
        "insidehighered": "mds-surface",
        "hercjobs": "bti-job-detail-pane",
        "timeshighereducation": "js-job-detail",
        "linkedin": "decorated-job-posting__details",
        "careerjet": "container",
    }

    domain = get_domain(url)

    got_job = False
    retry_times = 3
    for i in range(retry_times): #try 3 times to get the page
        async def get_page(): #startin up the pyppeteer
            browser = await launch()
            page = await browser.newPage()
            await stealth(page) #lookin at you indeed
            await page.goto(url)
            page = await page.content()
            await browser.close()
            return page
        try:
            page = asyncio.get_event_loop().run_until_complete(get_page()) #get the page raw output
            got_job = True
            break
        except PyppeteerError as e:
            cprint(f"Error: in Pyppeteer occurred: {e}\nTrying {retry_times-i} more times", "magenta")
            time.sleep(1) #sleeep for a second if we fail to get the page
        
    if not got_job:
        return False

    soup = BeautifulSoup(page,features="lxml") #start up beautifulsoup

    #remove script and style tags, sometimes since we're getting the text out of all the tags, these get mixed in sometimes and give a lot of bad data
    soup = bs_remove_tags(soup, ['script','style'])
    
    #need to do some per-site cleaning to remove extra text:
    if domain == "hercjobs":
        for div in soup.find_all("div", {'class':'d-print-none'}): 
            div.decompose()
        for button_tags in soup.select('button'): 
            button_tags.extract()
        for div in soup.find_all("div", {'modal'}): 
            div.decompose()

    if domain == "insidehighered" or domain == "chronicle":
        for div in soup.find_all("div", {'mds-text-align-right'}):
            div.decompose()
        for div in soup.find_all("div", {'mds-border-top'}):
            div.decompose()

    if domain == "timeshighereducation":
        for ul in soup.find_all("ul", {'job-actions'}):
            ul.decompose()
        for div in soup.find_all("div", {'float-right'}):
            div.decompose()
        for div in soup.find_all("div", {'premium-disabled'}):
            div.decompose()
        for span in soup.find_all("span", {'hidden'}):
            span.decompose()
        for div in soup.find_all("div", {'job-sticky-ctas'}):
            div.decompose()
    if domain == "careerjet":
        soup = bs_remove_tags_by_class(soup, 'section', 'fwd')
        soup = bs_remove_tags_by_class(soup, 'section', 'actions')
        soup = bs_remove_tags_by_class(soup, 'section', 'nav')
        soup = bs_remove_tags_by_class(soup, 'p', 'source')

    if domain in jobsite_container_class.keys(): #see if we see the current domain in our list of domains that we have classes for
        elements = soup.find_all("div", class_ = jobsite_container_class[domain]) #find the container
        try:
            text = elements[0].get_text(separator=' ') #get text but add spaces around elements
        except IndexError:
            text = soup.get_text(separator=' ') #if we don't find the container, we'll just get all the text, and throw an error
            cprint(f"NOTE: {domain}'s container div wasn't found. might want to read the get_jorb function", "magenta")
    else:
        print(f"NOTE: did not recognize jobsite {domain}, consider adding to get_jorb function")
        text = soup.get_text(separator=' ') #we'll just get all the text, and throw an error

    
    text = html.unescape(text) #fixes html escaped text, like the &nbsp; nonbreaking space
    text = re.sub('[^\S\r\n]+', ' ', text) #remove doubled spaces
    text = re.sub('\n+\s*\n+','\n', text) #remove extra linebreaks
    text = re.sub('\n\s+','\n', text) #remove spaces at the start of lines
    
    if domain == "linkedin":
        text = text.split("Show more\nShow less")[0]

    return text

def bs_remove_tags(soup, tags):
    for tag in tags:
        for match in soup.findAll(tag):
            match.replaceWithChildren()
    return soup

def bs_remove_tags_by_class(soup, tag, class_name):
    for item in soup.find_all(tag, {class_name}):
        item.decompose()
    return soup

def write_jorb_csv_log(output,timestamp): #this is our output file 
    filename = f"collected_jorbs_{timestamp}.csv"
    if not os.path.exists(filename):#if the output file doesn't exist yet we'll add a header
        header = list(output.keys())

        #this header is coming directly from the output ,we'll clean the output a bit
        header.sort() 
        header.remove('1_date_time')
        header.remove('2_job_link')
        header = ['job_link'] + header
        header = ['date_time'] + header
    
        
        with open(filename, 'a') as f_object: #write the csv header
            writer_object = writer(f_object)
            writer_object.writerow(header)
            f_object.close()

    #write the output line to the csv
    with open(filename, 'a') as f_object: #write the output as a csv
        writer_object = writer(f_object)
        writer_object.writerow(output.values())
        f_object.close()


#reference https://blog.daniemon.com/2023/06/15/chatgpt-function-calling/
def gpt_jorb(jorb,open_ai_key,functions,field_name,relevance):

    #do the chat gpt function request
    try:
        openai.api_key = open_ai_key
        response = openai.ChatCompletion.create(
            #model = "gpt-3.5-turbo-0613",
            #model = "gpt-4-0613",
            model = "gpt-4-1106-preview",
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at reading job descriptions, and determing the kind of job, job requirements, job salary, job location, and other job information."
                },
                {
                    "role": "user",
                    "content": jorb
                }
            ],
            functions = functions,
            function_call = {
                "name": functions[0]["name"]
            }
        )

        #gets the structured response
        arguments = response["choices"][0]["message"]["function_call"]["arguments"]


        arguments = re.sub("\n", ' ', arguments) #remove linebreaks inside the json fields,
        arguments = re.sub("\s+", ' ', arguments) #remove doubled spaces in the json
        
        #parses the structured response
        arguments = json.loads(arguments)
        
        for key in functions[0]['parameters']['required']: #make sure that if chatgpt fails to return a value we add it in so the output csv isn't a mess
            if not key in arguments:
                arguments[key] = "null"

        #we're going to do a followup question for chatgpt to try and figure out if the job is actually relevant to our interests
        try:
            prompt = f"{relevance}\n\nJob Title:\n{arguments['job-title']}\n\nJob Description:\n{arguments['summary']}\n\nJob Requirements:\n{arguments['requirements']}" #build our prompt

            openai.api_key = open_ai_key
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
            reply = completion.choices[0].message.content #just get the reply message which is what we care about

            #hopefully we get a true or false, but sometimes chatgpt does something weird, so we'll try to clean it up
            reply = re.sub("\n", ' ', reply) #remove linebreaks
            reply = re.sub(r'\s.+', '', reply) #seems like mostly when chatgpt does something weird here, it explains why it's true or false, we'll just drop everything after a space (ie "FALSE. This job is not..." to "FALSE.")
            reply = re.sub(r'[^a-zA-Z]', '', reply) #drop all non alpha from string

            arguments[field_name]=reply
        except:
            return False
        arguments = OrderedDict(sorted(arguments.items()))#sort our output
        return arguments
    except:
        return False


#text message function
def send_text(phone,message,api_key):
    try:
        resp = requests.post('https://textbelt.com/text', {
        'phone': phone,
        'message': message,
        'key': api_key,
        })
        return True
    except:
        return False
    

def get_linkedin_search(url):
    async def get_feed_raw():  # startin up the pyppeteer
        browser = await launch()
        page = await browser.newPage()
        await page.setViewport({"width": 3048, "height": 2160}) #bigger screen size seemed to make for less scrolling
        await stealth(page)  # lookin at you indeed
        await page.goto(url)

        if len(await page.content()) < 100: #if the page is less than 100 characters, something went wrong, so we'll return 0
            cprint("ERROR: the linkedin page didn't load", "magenta")
            return 0

        #check to see if there are any jobs, for whatever reason, linked in sometimes uses different text for this
        no_results_texts = ["No matching jobs found.","Please make sure your keywords are spelled correctly"]
        for no_results_text in no_results_texts:
            is_text_visible = await page.evaluate('''() => {
                return document.body.innerText.includes("No matching jobs found.");
            }''')
            if is_text_visible:
                return 0
        
        click_failed = 0

        scrolls = 1500
        time_of_scrolls = int(time.time())
        current_second = time_of_scrolls
        for i in range(scrolls): #i had a while true here, but it's better if it fails eventually
            if (int(time.time())-time_of_scrolls)%30 == 0 and not current_second == int(time.time()): #if we've been waiting for for a while, lets give an update
                print(f"      Linkedin still scrolling after {round(time.time()-time_of_scrolls)} seconds. {i}/{scrolls} scrolls left")
                current_second = int(time.time())
            


            #wait for the loader to disappear
            #print(f"i:{i}")
            for j in range(100):
                #print(f"j:{j}")
                is_visible = await page.evaluate(
                    """() => {
                    let loader = document.querySelector('.loader');
                    if (!loader) return false;
                    let style = window.getComputedStyle(loader);
                    return style.display !== 'none';
                }"""
                )
                if not is_visible:
                    break  # If the loader is not visible, break the loop
                else:
                    await asyncio.sleep( rand_sleep() ) #if the loader is visible sleep for a short random time

            #scroll the page by the page height
            await page.evaluate("window.scrollBy(0, window.innerHeight)")

            #going to try and click the show more button, but it isn't visible until you've scrolled a long ways, so we'll keep trying until we can click it
            try:
                await asyncio.sleep( rand_sleep() )  #Sleep before we click for a short random time
                await page.click(".infinite-scroller__show-more-button") #try clicking, if it fails, we bounce to the except below, otherwise we break out of the loop
            except:
                click_failed += 1

            #this text appears at the bottom of the results, if we see it we break
            is_text_visible = await page.evaluate(
                """() => {
                return document.body.innerText.includes("You've viewed all jobs for this search");
            }"""
            )
            if is_text_visible:
                break  # If the element is found, break the loop

        if i > (scrolls-5):
            cprint(f"Warning: We scrolled {scrolls} times and didn't find the end of the page, maybe make your keyword more specifi cor open the link to see if there's something else going on.","BLUE")


        feed_raw = await page.content() #get the page content
        await browser.close()
        return feed_raw #to the try below

    try:
        feed_raw = asyncio.get_event_loop().run_until_complete(
            get_feed_raw()
        )  
    except:
        cprint("ERROR: get_feed_raw failed", "magenta")
        return False

    if len(str(feed_raw)) > 100: #sometimes feed_raw comes out as an int?
        #start up beautifulsoup parser
        soup = BeautifulSoup(feed_raw, "html.parser")
        divs = soup.find_all("div", attrs={"data-entity-urn": True}) #linked in keeps the job id in the div attributes, so we'll grab that

        jobs = []
        for div in divs:
            job_id = div["data-entity-urn"].split(":")[-1] #get the job id, in the site it looks like "data-entity-urn="urn:li:jobPosting:3755470964", we need that number at the end
            jobs.append(f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}")
        print(f"      Found    {len(jobs)} jobs on linkedin")#show the jobs count

        if(len(jobs) == 0): #if we didn't find any jobs, we'll return 0
            return 0
        return jobs
    else:
        cprint("ERROR: get_feed_raw seemed to work, but didn't return full results", "magenta")
        print(f"feed_raw: {feed_raw}")
        return False

#get the domain by itself ie "https://www.asdf.com/qwerty" returns just "asdf"
def get_domain(url): 
    domain = tldextract.extract(url)
    domain = domain.domain
    return domain

def rand_sleep():
    return random.uniform(0.01, 0.05)



#https://github.com/careerjet/careerjet-api-client-python didn't use their librarysince it's just a simple request, but this has info about the api
def get_careerjet_jobs(search_term, page):
    url = "http://public.api.careerjet.net/search"
    params = {
        'locale_code': 'en_US',
        'keywords': search_term,
        'user_ip': '192.168.1.1',
        'user_agent': 'python3.8',
        'pagesize': '100',
        'page':{page},
    }
    response = requests.get(url, params=params)
    return response.json()

#the json returns a link that has a redirect, but it turns out it just swaps some bits around, so we generate that link here to save time on requests later
def careerjet_convert_url_to_page(url):
    # Split the URL by periods and slashes
    parts = re.split('[./]', url)
    # Get the second to last item
    second_last_item = parts[-2] if len(parts) > 1 else None
    return f"https://www.careerjet.com/jobad/us{second_last_item}"


def get_careerjet_job_links(search_term):
    #get the first page of results
    response = get_careerjet_jobs(search_term, 1)

    #make sure that there are jobs in the response
    if response['hits'] >= 1:
        pages = response['pages']
        output = []

        #collect the jobs from the first page
        for job in response['jobs']: 
            output.append(careerjet_convert_url_to_page(job['url']))

        #collect the jobs from the remaining pages
        for i in range(2, pages):
            for job in response['jobs']:
                output.append(careerjet_convert_url_to_page(job['url']))
        
        feed_domain = get_domain (output[0])

        print(f"      Found    {len(output)} jobs on {feed_domain.capitalize()}")
        return output
    else:
        return 0
    