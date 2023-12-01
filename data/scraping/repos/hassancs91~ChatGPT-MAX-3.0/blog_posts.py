import newspaper
from langchain.text_splitter import TokenTextSplitter
import prompts


def split_text_into_chunks(text, max_tokens):
    text_splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks


def get_article_from_url(url):
    try:
        # Scrape the web page for content using newspaper
        article = newspaper.Article(url)
        # Download the article's content with a timeout of 10 seconds
        article.download()
        # Check if the download was successful before parsing the article
        if article.download_state == 2:
            article.parse()
            # Get the main text content of the article
            article_text = article.text
            return article_text
        else:
            print("Error: Unable to download article from URL:", url)
            return None
    except Exception as e:
        print("An error occurred while processing the URL:", url)
        print(str(e))
        return None


def get_blog_summary_prompt(blog_url):
    # https://learnwithhasan.com/chatgpt-earthquake/
    blog_article = get_article_from_url(blog_url)
    prompt = prompts.blog_bullet_summary_prompt.format(
        MaxPoints="10", MinPoints="5", InputText=blog_article
    )
    return prompt
