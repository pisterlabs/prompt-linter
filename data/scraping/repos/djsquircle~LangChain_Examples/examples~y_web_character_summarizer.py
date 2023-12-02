import os
import json
import requests
from dotenv import load_dotenv
from newspaper import Article
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from typing import Optional


def generate_character_profile_summary(url: str, model_name: Optional[str] = "gpt-3.5-turbo", temperature: Optional[float] = 0) -> str:
    """
    Generate a character profile summary from a webpage.

    Args:
        url: The URL of the webpage with the character profile.
        model_name: The name of the OpenAI model to use.
        temperature: The temperature parameter controlling the randomness of the output.

    Returns:
        A summary of the character profile in bulleted list format.
    """
    
    # Load environment variables if a .env file exists
    if os.path.exists(".env"):
        load_dotenv()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    
    session = requests.Session()

    try:
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            article = Article(url)
            article.download()
            article.parse()

            # Template for summarizing the character profile
            template = """
            Please summarize the character profile from the article below:

            ==================
            Title: {article_title}
            {article_text}
            ==================

            Provide a summarized version in bulleted points.
            """

            # Format the prompt
            prompt = template.format(article_title=article.title, article_text=article.text)

            messages = [HumanMessage(content=prompt)]

            # Load the model
            chat = ChatOpenAI(model_name=model_name, temperature=temperature)

            # Generate summary
            summary = chat(messages)
            return summary.content
        
        else:
            return f"Failed to fetch article at {url}"
    except Exception as e:
        return f"Error occurred while fetching article at {url}: {e}"

# Example usage
if __name__ == "__main__":
    url = "https://daogen.ai/falcon-a-quest/"
    summary = generate_character_profile_summary(url)
    print(summary)
