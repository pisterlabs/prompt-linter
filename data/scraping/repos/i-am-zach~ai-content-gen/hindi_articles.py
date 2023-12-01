from prefect import task, flow, get_run_logger
import datetime as dt
from rss_feed import RSSFeedProvider
from pathlib import Path
import os
import json
from selenium import webdriver
from slugify import slugify

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv

from article_scraper import scrape_india_today, scrape_the_hindu, scrape_ndtv, scrape_toi

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out")
OUTPUT_PARSER = CommaSeparatedListOutputParser()
DATE_HASH = lambda: dt.date.today().strftime("%Y-%m-%d")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@task
def fetch_and_dump_provider(provider: RSSFeedProvider, output_dir: Path, logger):
    logger.info(f"Fetching {provider.id} feed")
    feed = [{ 'provider': provider.id, **article.to_dict()} for article in provider.get_feed()]
    output_dir = Path(OUTPUT_DIR) / 'feeds' / DATE_HASH()
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / f"{provider.id}.json", 'w') as f:
        logger.info(f"Dumping {provider.id}.json to file")  
        json.dump(feed, f)
    return feed


@flow
def get_rss_feed() -> list:
    """Get the rss feeds from the india news providers and dumps them to a json file"""
    logger = get_run_logger()
    output_dir = Path(OUTPUT_DIR) / 'feeds' / DATE_HASH()
    rss_feeds = []
    for provider in RSSFeedProvider.__subclasses__():
        if os.path.exists(output_dir / f"{provider.id}.json"):
            logger.info(f"Found {provider.id}.json. Loading from file")
            with open(output_dir / f"{provider.id}.json", 'r') as f:
                feed = json.load(f)
        else:
            logger.info("Could not find {provider.id}.json. Fetching from web")
            try:
                feed = fetch_and_dump_provider(provider, output_dir, logger)
            except Exception as e:
                logger.warning(f"Failed to fetch {provider.id} feed. Error: {e}")
                feed = []

        rss_feeds.extend(feed)

    return rss_feeds

@task
def stringify_articles(articles: list) -> str:
    return "\n".join([f"{i+1}. {article['title']}" for i, article in enumerate(articles)])
        
@task
def get_interesting_articles_prompt(articles_list_string: str):
    prompt = PromptTemplate(
    template="""You are a news editor and you are looking for interesting articles to write about. Your newspaper is called "Hindi88.fm" and you educate Indian americans about current indian news.

    One of your writers has gone through some websites looking for interesting articles. He has given you the following list of article titles. You want to read the article titles and decide which ones are worth writing a story about. An article is worth writing a story about if it satisfies the following criteria:
    1) Relevant to your newspaper. If the articles are not relevant, you will not read them.
    2) Positive outlook. Your newspaper is not a tabloid, so you will not read articles that are depressing or negative.
    3) Interesting to your readers. Your readers are Indian americans, so you will not read articles that are not interesting to them.

    ---
    List of article titles:
    {article_titles_list}
    ---

    You will read the article titles and decide which ones are relevant, positive, and interesting. Return ONLY the ARTICLE NUMBERS in your response and nothing else.\n{format_instructions}.
    """,
        input_variables=["article_titles_list"],
        partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()}
    )
    return prompt.format(
        article_titles_list=articles_list_string
    )

@task
def fetch_interesting_articles(prompt: str, articles: list) -> list:
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    output = llm(prompt)
    indexes = [int(i) for i in OUTPUT_PARSER.parse(output)]
    return [articles[i-1] for i in indexes]

@task
def load_interesting_articles(interesting_articles_path: Path):
    return json.load(open(interesting_articles_path))

@task
def store_interesting_articles(interesting_articles: list, interesting_articles_path: Path):
    with open(interesting_articles_path, 'w') as f:
        json.dump(interesting_articles, f)

@task
def scrape_article(driver, provider: str, url: str) -> list[str]:
    if provider == "indiatoday":
        return scrape_india_today(driver, url)
    elif provider == "ndtv":
        return scrape_ndtv(driver, url)
    elif provider == "thehindu":
        return scrape_the_hindu(driver, url)
    else:
        raise NotImplementedError(f"Provider {provider} not implemented")


@task
def store_scraped_interesting_article_content(lines: list[str], provider: str, title: str):
    output_dir = Path(OUTPUT_DIR) / 'articles' / provider
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir / f"{slugify(title)}.txt", 'w') as f:
        f.write('\n'.join(lines))

@task
def summarize_article(provider: str, title: str, lines: list[str]):
    prompt = PromptTemplate(
        template="""You are a news editor and you are looking for interesting articles to write about. Your newspaper is called "Hindi88.fm" and you educate Indian americans about current indian news.
        
        You have found an interesting article to write about. The article is about {title} from {provider}.         

        You want to use the article to write your own article in the style of the Morning Brew. You want to write an article that is easy to read and understand. You want to write an article that is interesting. 

        ---
        Original article:
        {content}

        ---

        Instructions: The first line of the article should be the title. The rest of the article should be the content.

        Your Article:\n""",
        input_variables=["title", "provider", "content"],
    )
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    article_summary_prompt = prompt.format(
        title=title,
        provider=provider,
        content='\n'.join(lines)
    )
    output = llm(article_summary_prompt)
    return output


@flow
def interesting_articles_flow():
    logger = get_run_logger()
    rss_feed_articles = get_rss_feed()

    interesting_articles_prompt = get_interesting_articles_prompt(stringify_articles(rss_feed_articles))

    interesting_articles_path = Path(OUTPUT_DIR) / 'interesting_articles' / f"{DATE_HASH()}.json"
    if not os.path.exists(interesting_articles_path):
        interesting_articles = fetch_interesting_articles(interesting_articles_prompt, rss_feed_articles)
        store_interesting_articles(interesting_articles, interesting_articles_path)
    else:
        interesting_articles = load_interesting_articles(interesting_articles_path)

    for interesting_article in interesting_articles:
        provider = interesting_article['provider']
        title_slug = slugify(interesting_article['title'])
        url = interesting_article['link']
        if os.path.exists(Path(OUTPUT_DIR) / 'articles' / provider / f"{title_slug}.txt"):
            logger.info(f"{interesting_article['provider']} article {interesting_article['title']} already scraped. Skipping")
            continue

        # Check if the article is already scraped
        if os.path.exists(Path(OUTPUT_DIR) / 'articles' / provider / f"{title_slug}.txt"):
            continue
        if provider == "timesofindia":
            continue
        try:
            driver = webdriver.Safari()
            scraped_article_lines = scrape_article(driver, provider, url)
            store_scraped_interesting_article_content(scraped_article_lines, interesting_article['provider'], interesting_article['title'])
        except Exception as e:
            logger.warning(f"Failed to scrape {interesting_article['provider']} article {interesting_article['title']} with url {interesting_article['link']}. Error: {e}")
        finally:
            driver.quit()


@flow(log_prints=True)
def summarize_article_basic_flow():
    with open("out/articles/indiatoday/india-s-g20-presidency-the-emerging-global-order.txt") as f:
        lines = f.readlines()

    title = "India's G20 presidency | The emerging global order"
    provider = "indiatoday"
    article_summary_output = summarize_article(provider, title, lines)
    print(article_summary_output)
    os.makedirs(Path(OUTPUT_DIR) / 'summaries', exist_ok=True)
    with open(Path(OUTPUT_DIR) / 'summaries' / f'{slugify(title)}.txt', 'w') as f:
        f.write(article_summary_output)




if __name__ == "__main__":
    interesting_articles_flow.serve(name="interesting_articles_deployment")
    # summarize_article_basic_flow.serve(name="summarize_article_basic_deployment")
