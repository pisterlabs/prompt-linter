import openai
import sys
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin
import tiktoken
import json
import prompts
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import psutil

def scrape_website(website_url):
    posts = {}
    def extract_links(url, existing_hrefs=set(), max_depth=4, depth=0):
        
        base_url = urlparse(url).scheme + "://" + urlparse(url).netloc
        response = requests.get(url)
        
        if response.status_code == 200:
            # valid url, add to existing_hrefs
            existing_hrefs.add(url)

            # get page content and save to posts for classification later on
            soup = BeautifulSoup(response.content, 'html.parser')
            posts[url] = soup.get_text()

            if(depth == max_depth):
                return existing_hrefs

            # TODO: Add post classification, title extraction, and date extraction here

            # get links inside the page soup
            found_hrefs = set()
            for element in soup.find_all(href=True):
                href_value = element['href']
                
                # if href_value is relative, make it absolute. If it's an external link, it will be unchanged
                absolute_url = urljoin(base_url, href_value)

                # only add links from same domain
                if urlparse(absolute_url).netloc == urlparse(url).netloc:
                    found_hrefs.add(absolute_url)
            

            # determine which links are new from this page
            new_hrefs = found_hrefs.difference(existing_hrefs)

            # if there are new links, explore them to determine if more links exist
            if len(new_hrefs) > 0:

                # update existing_hrefs with new_hrefs so we don't explore new_hrefs in any subsequent recursive calls
                existing_hrefs = existing_hrefs.union(new_hrefs)

                # for each link in new_hrefs, let's recurisvely explore it to see if there are more pages to find
                for href in new_hrefs:
                    new_existing_hrefs = extract_links(href, existing_hrefs, max_depth, depth + 1)

                    # add any new links found in the recursive call to the existing_hrefs set
                    existing_hrefs = existing_hrefs.union(new_existing_hrefs)
                
        else:
            # Invalid URL, so remove the URL from the list of links to scrape
            existing_hrefs.remove(url)
        
        # return all the links we found
        return existing_hrefs

    return posts, extract_links(website_url, max_depth=4) 

def write_posts_to_disk(filename, posts):
    with open(f'./data/{filename}.json', 'w') as fp:
        json.dump(posts, fp, indent=4)

def extract_page_classification(url, body):
    # Limit the number of tokens for input to
    prompt = prompts.BLOGPOST_CLASSIFICATION_AND_EXTRACT_TITLE_DATE
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo") 

    num_tokens = len(enc.encode(body + prompt + url))
    if num_tokens > 4000:
        num_other_tokens = len(enc.encode(prompt + url))
        body = enc.decode(enc.encode(body)[:4000 - num_other_tokens])

    # get the chatGPT response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt} \n \n URL: {url} \n Body: {body}"},
            ]
    )
    classification = response["choices"][0]["message"]["content"]

    try:
        return json.loads(classification)
    except:
        raise ValueError(f"Unable to parse chatGPT response. The following: \n \n {classification} \n \n is not JSON")


if __name__ == "__main__":
    website_url = sys.argv[1]
    
    # step 1 - scrape website
    pid = os.getpid()
    posts, all_links = scrape_website(website_url)
    print("Memory usage: ", psutil.Process(pid).memory_info().rss / 1024 ** 2, "MB")


    # # write posts to disk   
    # filename = urlparse(website_url).netloc.replace('/', '_').replace(':', '_').replace('.', '_')
    # write_posts_to_disk(filename, pages)


    # extract page classification
    # classifications = []
    # start_time = time.time()
    # for i, url in zip(range(len(posts)), posts):
    #     print(f"Classifying {i}/{len(posts)}: {url}")

    #     elapsed_time = time.time() - start_time
    #     print(f"Elapsed time: {elapsed_time} seconds")

    #     executor = ThreadPoolExecutor(max_workers=1)

    #     for attempt in range(10):
    #         future = executor.submit(extract_page_classification, url, posts[url])
    #         try:
    #             jsonresponse = future.result(timeout=20)
    #             classifications.append(jsonresponse)
    #             break
    #         except:
    #             if attempt > 9:  # 0-indexed, so 9 is the 10th attempt
    #                 raise TimeoutError(f"Timed out on {url}")
    #             print(f"Failed Attempt {attempt+1} of 10 on {url}. Waiting 60 seconds...")
    #             time.sleep(60)



    # # write classifications to disk
    # with open ('data/pg_classifications.json', 'w') as fp:
    #     json.dump(classifications, fp, indent=4)

# for link in all_links:
#     print(link)

# all_links = list(all_links)

# # download all text in links as a dictionary
# text_dict = {}
# for link in all_links:
#     response = requests.get(link)
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.content, 'html.parser')
#         text_dict[link] = soup.get_text()[:2000]
#     else:
#         print(f"Failed to fetch the URL: {link}")

# token_counts = []
# enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
# for link in text_dict:
#     num_tokens = len(enc.encode(text_dict[link]))
#     token_counts.append(num_tokens)

# print("Average number of tokens per link:", sum(token_counts) / len(token_counts))
# print("Total number of tokens:", sum(token_counts))


# # step 3: Classify links into articles or non articles
# # use this prompt from ChatGPT:
# PROMPT = '''I need you to act as a URL classifier. You are going to guess whether the given URL is an article based on the URL only.  Give me your best guess. If you believe the page is a written article, please output the URL with yes (like so: "[url]: yes"). If you think it contains a collection of articles, or the homepage of a blog, or anything that is NOT an article, output the url with no (like this: "[url]: no"). Do not respond with anything else except the URLs and the "yes" or "no".'''
# NUM_LINKS_TO_PROCESS = 50

# for i in range(len(all_links)):
#     num_links_in_call = min(NUM_LINKS_TO_PROCESS, len(all_links) - i)
#     included_links = all_links[i:i+num_links_in_call]
#     included_links_string = ""
#     for link in included_links:
#         included_links_string += link
#         included_links_string += " \n "

#     # get openAI predictions
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": f"{PROMPT} \n \n {included_links_string}"},
#             ]
#     )
    
    

    # print(response["choices"]["message"]["content"])

    # i += num_links_in_call




# step 4: Parse data from HTML on website to get blog text
