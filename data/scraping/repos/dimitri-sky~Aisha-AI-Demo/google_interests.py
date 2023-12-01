import asyncio
import re
import requests
import openai
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from logger import setup_logger
from keys import OPENAI_KEY, GOOGLE_SEARCH_API, GOOGLE_SEARCH_ID

# Set up logger
logger = setup_logger('google_interests')

# Set up OpenAI API
openai.api_key = OPENAI_KEY


def google_search(query, api_key, cse_id, **kwargs):
    '''Perform Google search and return results'''
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()

    if 'items' in res:
        return res['items']
    else:
        logger.error(
            "Google Search response did not contain 'items'. return None.")
        return None


def clean_data(data):
    '''Clean up the data (remove new lines and trim whitespaces)'''
    return [i.replace('\n', ' ').strip() for i in data]


def get_content(url):
    '''Function to scrape and return contents from a specific URL'''
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
        }
        page = requests.get(url, headers=headers)

        if page.status_code == 200 and "cloudflare" not in page.text.lower() and "twitter.com" not in url:
            # Extract and clean content from specific tags
            soup = BeautifulSoup(page.content, 'html.parser')
            tags = ['h1', 'h2', 'p', 'a', 'div']
            contents = {tag: clean_data(
                [i.text for i in soup.find_all(tag)]) for tag in tags}
            return contents
        else:
            logger.warning(f"Could not scrape content from {url}.")
            return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


async def get_most_relevant_link_from_GPT_non_blocking(system_message, n=0):
    '''Non-blocking version of function to get relevant link from GPT'''
    loop = asyncio.get_event_loop()
    task = loop.create_task(get_most_relevant_link_from_GPT(system_message))
    return await task


async def get_most_relevant_link_from_GPT(system_message, n=0):
    '''Function to get the most relevant link from GPT based on system message'''
    messages = [
        {"role": "system", "content": system_message}
    ]
    try:
        response_data = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.4
        )

        return response_data.choices[0].message['content'].strip().strip('.').strip('"')
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


async def summarize_google(user_input, user_profile, last_3_conversations):
    '''Main function to summarize news based on user input, profile, and conversation history'''
    api_key = GOOGLE_SEARCH_API
    cse_id = GOOGLE_SEARCH_ID

    system_message = f"Aisha, as an AI, you are about to decide what overall sentence to Google for the user. Considering the latest text message from the user: \"{user_input}\", the last 6 conversations: \"{last_3_conversations}\" and the user profile: \"{user_profile}\"\nWhat is the most relevant short google search, that most likely will have results, to continue this conversation? Google Search:"
    query = await get_most_relevant_link_from_GPT_non_blocking(system_message)
    logger.info(f"Query: {query}")

    # Perform Google search and process the results
    results = google_search(query, api_key, cse_id, num=9)

    if results is None:
        logger.error("No results from Google Search.")
        return "No results from Google Search.", None

    url_titles = [f"{i}. {result['title']}" for i,
                  result in enumerate(results)]

    logger.info(f"url_titles: {url_titles}")

    # Prepare prompt for GPT
    system_message = f"This is a list of the top 9 google search results for the query '{query}':\n"
    system_message += '\n'.join(url_titles)
    system_message += f"\nWhat is the one most relevant, amusing, and interesting news title number for the query '{query}', answer with only the number. \nNumber: "

    most_relevant_url = await get_most_relevant_link_from_GPT_non_blocking(system_message)

    # Using the number from the AI response, get the URL link from the results list
    try:
        most_relevant_url = re.match(r'^(\d+)', most_relevant_url).group()
    except:
        most_relevant_url = 1

    most_relevant_url = results[int(most_relevant_url)]["link"]
    logger.info(f"Most relevant URL (Google Search): {most_relevant_url}")

    content = get_content(most_relevant_url)

    # If the content is None, try all other URLs in the results
    if content is None:
        for i in range(len(results)):
            # Skip the URL that we've already tried
            if results[i]["link"] == most_relevant_url:
                continue
            logger.debug("CONTENT IS NONE, MOVING TO LINK: %s", i)
            content = get_content(results[i]["link"])
            if content is not None:
                most_relevant_url = results[i]["link"]
                break

    # If all URLs returned None (all were blocked or were twitter links), then return a message
    if content is None:
        logger.error(
            "All URLs were either blocked for scraping or were twitter links.\n")
        return "Could not fetch news. Please try again later.", None

    information = "\n".join(
        [f"{k}: {str(v)[:400]}" for k, v in content.items()])

    # Ask GPT to summarize the information
    system_message = f"Extract and summarize only news about \"{query}\" with dates from the URL: {most_relevant_url}\nInformation:\n{information}\nSummary:"

    summary = await get_most_relevant_link_from_GPT_non_blocking(
        system_message)  # + f" [Link: {most_relevant_url}]"
    logger.info(f"Summary: {summary}")

    return summary, most_relevant_url


if __name__ == '__main__':
    # Simple Test
    asyncio.run(summarize_google("Zoe Support",
                "Likes playing LOL", "I like playing LOL"))
