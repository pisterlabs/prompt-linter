import openai
import time
import random
from scrape_articles import scrape_articles
from datetime import datetime
import os 



openai.api_key_path = 'openai_keys.py'  # HIDE THIS BEFORE PUBLISHING


def process_articles():
    """ Takes content through a series of prompts that: deduplicate stories, determine their relevance to product management, and generate summaries for relevant stories. Also generates an intro
    for the email using the combined summaries and a daily prompt
    :return: dict
    """
    scraped_articles = scrape_articles()


    def filter_articles_topics(article_text):
        """ This sends the article text to GPT where it evaluates if the topic should be included. Relevant articles lead to a response of 'relevant', irrelevant articles lead to a
        response of 'irrelevant'. The articles with the 'irrelevant' tag are filtered out of the set of articles to include in the email.
        :param str article_text: This is the text we want to evaluate in terms of relevant topics
        :return: str
        """
        # Truncate the text to a certain length if necessary. This avoids sending content over the allowed token count and helps control cost
        max_length = 1500  # Adjust this value as needed
        if len(article_text) > max_length:
            article_text = article_text[:max_length]
            print(f"Article text too long. Truncated to {max_length} characters.")
        # Proceed with filtering articles
        for attempt in range(5):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",
                         "content": "Analyze the text provided to determine if the article should be included in our content based on the instructions I will provide. We are screening out anything in the following categories..\n\n"
          "Screen the text for primary topics related to space, automotive, crypto, argriculture, video games, specific video game consoles (xbox, playstation), or specific geographical regions (China or India). If these are the primary focus, respond with: 'irrelevant'. If not 'irrelevant' meaning it is not in one of these forbidden topics then respond with 'relevant'. Only respond with 'relevant' or 'irrelevant', do not return any other response."},
                        {"role": "user",
                         "content": article_text}
                    ],
                    max_tokens=300,
                    temperature=.8,
                )
                summary = response["choices"][0]["message"]["content"].strip()
                print(summary)
                return summary

            except openai.error.ServiceUnavailableError:
                wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                print(f"Server error, retrying in {wait_time} seconds")
                time.sleep(wait_time)
        print("Failed to generate response after several retries")
        return None
    def filter_articles_relevance(article_text):
        """ This sends the article text to GPT where it evaluates it for relevance to product management. Relevant articles lead to a response of 'product related', irrelevant artielcles lead to a
        response of 'not product related'. The articles with the 'not product related' tag are filtered out of the set of articles to include in the email.
        :param str article_text: This is the text we want to evaluate in terms of relevance to product management
        :return: str
        """
        # Truncate the text to a certain length if necessary. This avoids sending content over the allowed token count and helps control cost
        max_length = 1500  # Adjust this value as needed
        if len(article_text) > max_length:
            article_text = article_text[:max_length]
            print(f"Article text too long. Truncated to {max_length} characters.")
        # Proceed with filtering articles
        for attempt in range(5):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",
                         "content": "Analyze the text provided to determine its relevance to product management.\n\n"
          "- Consider the main theme of the article. Ask yourself: Does the theme relate to any aspect of software or technology product management, including but not limited to product development, product strategy, product design, product lifecycle, product marketing, stakeholder management, business models, innovation, or product metrics? Additionally, consider any broader connections to technology or business that might be relevant to product managers, "
                                    "respond with: 'product related'. By default assume that most technology topics like SaaS software or consumer software apps"
                                    "are related to product management\n"
          "- If the text is about engineering tools, releases, or other technical details, evaluate if the context implies its connection to product management decision-making or responsibilities. If it does, respond with 'product related'; if not, 'not product related'.\n"
          "- If none of the above conditions are met, and the text doesn't focus on product management or its related concepts, respond with: 'not product related'."},
                        {"role": "user",
                         "content": article_text}
                    ],
                    max_tokens=300,
                    temperature=.8,
                )
                summary = response["choices"][0]["message"]["content"].strip()
                return summary

            except openai.error.ServiceUnavailableError:
                wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                print(f"Server error, retrying in {wait_time} seconds")
                time.sleep(wait_time)
        print("Failed to generate response after several retries")
        return None

    def remove_duplicates(article_set):
        """ Many sources may cover the same story in a given day. When this happens we only want to include the story one time using one source. The URLs for all scraped articles are sent to GPT
        and using the keywords in the URL it determines which URLs represent unique stories. In cases where there are multiple sources for a story it selects what it believes to be just the most
        reputable source and returns that URL.
        :param str article_set: All the URLs for the scraped articles are combined into a string so they can be sent to GPT for evaluation
        :return: list
        """
        for attempt in range(5):
            try:
                # In this case I used gpt-3.5-turbo-16k because I needed to send all URLs at once so the needed token count was much higher. But this model is twice as expensive so where possible I
                # used gpt-3.5-turbo
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {"role": "system",
                         "content": "You will receive a set of URLs. Analyze the keywords within each URL to identify potential overlapping content. If multiple URLs seem to discuss the same topic based on shared keywords (for instance, if 3 URLs contain the terms 'microsoft' and 'teams'), choose only one URL, giving preference to the most reputable source based on general knowledge about the source's reputation. After your analysis, provide a comma-separated list of unique URLs that correspond to distinct topics. Your response should only be the list of URLs, without any additional text, line breaks, or '\n' characters."},
                        {"role": "user",
                         "content": article_set}
                    ],
                    max_tokens=10000,
                    temperature=.2,
                )
                deduped_urls = response["choices"][0]["message"]["content"].replace('\n', '').strip()  # GPT only returns things in string format. So though the prompt asks for a column
                # separated list, the list actually comes back as a string that you need to parse. On occasion GPT was appending a \n to each URL which caused the subsequent parsing and matching to
                # break. In the case that happens, this strips out the \n

                return deduped_urls
            except openai.error.ServiceUnavailableError:
                wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                print(f"Server error, retrying in {wait_time} seconds")
                time.sleep(wait_time)
        print("Failed to generate response after several retries")
        return None

    def secondary_dedupe(article_set):
        """ Many sources may cover the same story in a given day. When this happens we only want to include the story one time using one source. The URLs for all scraped articles are sent to GPT
        and using the keywords in the URL it determines which URLs represent unique stories. In cases where there are multiple sources for a story it selects what it believes to be just the most
        reputable source and returns that URL.
        :param str article_set: All the URLs for the scraped articles are combined into a string so they can be sent to GPT for evaluation
        :return: list
        """
        for attempt in range(5):
            try:
                # In this case I used gpt-3.5-turbo-16k because I needed to send all URLs at once so the needed token count was much higher. But this model is twice as expensive so where possible I
                # used gpt-3.5-turbo
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {"role": "system",
                         "content": "You will receive a set of URLs. Analyze the keywords within each URL to identify cases where a company appears in more than one URL. If multiple URLs seem to discuss the same company based on shared keywords (for instance, if 3 URLs contain the terms 'Microsoft' or 'Apple'), choose only one URL, giving preference to the most reputable source based on general knowledge about the source's reputation. Do the same for cases where a similar topic (such as self-driving cars) is covered by more than one URL. After your analysis, provide a comma-separated list of unique URLs that correspond to distinct topics. Your response should only be the list of URLs, without any additional text, line breaks, or '\n' characters."},
                        {"role": "user",
                         "content": article_set}
                    ],
                    max_tokens=10000,
                    temperature=.2,
                )
                deduped_urls = response["choices"][0]["message"]["content"].replace('\n', '').strip()  # GPT only returns things in string format. So though the prompt asks for a column
                # separated list, the list actually comes back as a string that you need to parse. On occasion GPT was appending a \n to each URL which caused the subsequent parsing and matching to
                # break. In the case that happens, this strips out the \n

                return deduped_urls
            except openai.error.ServiceUnavailableError:
                wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                print(f"Server error, retrying in {wait_time} seconds")
                time.sleep(wait_time)
        print("Failed to generate response after several retries")
        return None
  
    def summary_generator(article_text):
        """ This sends the text of each article to GPT to have a summary generated
        :param str article_text: This is the text of an article that has been deemed relevant to product management
        :return: str
        """
        # Truncate the text to a certain length if necessary. This avoids sending content over the allowed token count and helps control cost
        max_length = 2000  # Adjust this value as needed
        if len(article_text) > max_length:
            article_text = article_text[:max_length]
            print(f"Article text too long. Truncated to {max_length} characters.")
        # Proceed with generating the summary
        for attempt in range(5):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",
                         "content": "As an AI specializing in text analysis with a focus on product management, your job is to summarize the provided text. You should generate a one sentence summary "
                                    "for the text. This summary should first outline the topic of the article and it should then describe why this article is relevant to product managers. The summaries should have a casual and fun but informed tone"},
                        {"role": "user",
                         "content": article_text}
                    ],
                    max_tokens=200,
                    temperature=.7,
                )
                summary = response["choices"][0]["message"]["content"].strip()
                return summary
            except openai.error.ServiceUnavailableError:
                wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                print(f"Server error, retrying in {wait_time} seconds")
                time.sleep(wait_time)
        print("Failed to generate response after several retries")
        return None

    def create_email_intro():
        """ The summaries for all the articles to be included in the email are passed into here and combined with a theme unique to the day of the week. These are sent to GPT where it creates a
        intro for the email.
        :return: str
        """
        current_date = datetime.now()
        day_of_week = current_date.weekday()
        # Unique theme for each day of the week
        if day_of_week == 0:
            theme = 'Encourage your readers to start the week with a positive, energized mindset. Lets get after it this week'
        elif day_of_week == 1:
            theme = 'Discuss unexpected turns and surprises in the field of product management.'
        elif day_of_week == 2:
            theme = 'Get through hump day with some witty banter and clever insights. Crack open an energy drink and power up to get through the week'
        elif day_of_week == 3:
            theme = 'Stimulate your neurons with some brain-teasing content. Maybe crack open an energy drink'
        elif day_of_week == 4:
            theme = 'Explore future technologies that could disrupt the field of product management.'
        elif day_of_week == 5:
            theme = 'Ride the wave of knowledge and insights from the week. Do something daring'
        else:
            theme = 'Use today to rest so you can be more productive the rest of the week. But be sure to stay up to date with what is happening by reading the newsletter. Sip some tea ' \
                    'while you relax'
        system_prompt = f"Hey there, AI! You're helping out with an intro for a daily product management newsletter called 'The PM A.M. Newsletter'. Today's theme is '{theme}'—pretty cool, " \
                        f"right? Don't mention the theme name " \
                        f"directly but instead incorporate it into the vibe of the overall intro. You'll get to chew on different " \
                        f"topics tied to the theme. Your job is to whip up an engaging and fun intro that'll give the readers a taste of what's to come in the newsletter. Don't get too caught up in the details of each article—just give a general vibe of the content ahead. And hey, feel free to drop in a casual joke or use emojis when it fits. We're all about keeping our readers alert and on their toes! Remember, the goal is to make product managers feel like they're kicking back with a can of knowledge that'll help them crush it in their work week. Let's make this exciting! Limit the intro to two sentences in length"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system",
                 "content": system_prompt},
                {"role": "user",
                 "content": intro_text}
            ],
            max_tokens=200,
            temperature=.7,
        )
        summary = response["choices"][0]["message"]["content"].strip()
        return summary

    # Create a list of unique URLs
    url_set = [f'{scraped_articles[key]["url"]},' for key in scraped_articles]
    deduped_articles = [url.strip() for url in remove_duplicates(str(url_set)).split(',')]

    # Remove articles that are not in deduped list
    for key in list(scraped_articles.keys()):
        if key not in deduped_articles:
            del scraped_articles[key]
    
    # Create a list of unique URLs after first dedupe
    url_set = [f'{scraped_articles[key]["url"]},' for key in scraped_articles]
    deduped_articles = [url.strip() for url in secondary_dedupe(str(url_set)).split(',')]

    # Remove articles that are not in the second deduped list
    for key in list(scraped_articles.keys()):
        if key not in deduped_articles:
            del scraped_articles[key]

    #Remove articles that are on unrelated topics
    keys_to_delete = []
    for key in scraped_articles.keys():
        relevance = filter_articles_topics(scraped_articles[key]['text'])
        if 'irrelevant' in relevance.lower():
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del scraped_articles[key]
    
    # Filter articles based on relevance
    keys_to_delete = []
    for key in scraped_articles.keys():
        relevance = filter_articles_relevance(scraped_articles[key]['text'])
        if 'not product related' in relevance.lower():
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del scraped_articles[key]

    # Generate summaries for relevant articles
    for key in scraped_articles.keys():
        summary = summary_generator(scraped_articles[key]['text'])
        scraped_articles[key]['summary'] = summary

    print(scraped_articles)
    print('finished')
    intro_text = []
    for key in scraped_articles:
        intro_text.append(scraped_articles[key]['summary'])

    intro_text = str(intro_text)

    # Generate intro for the email
    email_intro = create_email_intro()

    return scraped_articles, email_intro


if __name__ == "__main__":
    articles, intro = process_articles()
    for key, value in articles.items():
        print(f"{value['title']}({value['url']}):\n{value['summary']}")
    print(intro)



