import json
import os
import openai
from metaphor_python import Metaphor
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))

class BlogPostGenerator:
    def __init__(self, product_name, category):
        self.product_name = product_name
        self.category = category
        self.trending_articles = self.get_trending_articles()

    def get_trending_articles(self):
        # Using Metaphor API to find trending articles related to the product category
        query = f"Here are the Trending articles in {self.category} category"
        search_response = metaphor.search(
            query, use_autoprompt=True, start_published_date="2023-06-01", num_results = 3
        )

        trending_articles = search_response.results

        return trending_articles

    def find_similar_articles(self, article_url_list, num_results=3):
        # Use Metaphor API to find similar articles to a given article URL
        similar_articles_blog_content = "\n\n---\n\n**Similar Articles:**\n\n"

        for item in article_url_list:
            similar_articles_response = metaphor.find_similar(item, num_results=num_results)
            for result in similar_articles_response.results:
                similar_articles_blog_content += f"- [{result.title}]({result.url})\n"

        return similar_articles_blog_content

    def generate_intro(self, content):
        intro_prompt = f"This is the content of three articles separated by  ******  :   {content}. Based on this , generate a short introduction of 100 words. Add an appropriate topic to the first line. This is the starting part of a blog"
        return intro_prompt

    def generate_technical_specifications(self, content):
        specs_prompt= f"This is the content of three articles separated by  ******  :   {content}. Based on this , generate a body for a blog along with technical specifications in about 500 words"
        return specs_prompt

    def generate_conclusion(self, content):
        conclusion_prompt= f"This is the content of three articles separated by  ******  :  {content}. Based on this , generate a conclusion for a blog that is a combination of the given three articles in about 100 words"
        return conclusion_prompt



    def generate_blog_post(self, content):

        # Combining various sections to form a complete blog post
        intro = self.generate_intro(content)
        specs = self.generate_technical_specifications(content)
        conclusion = self.generate_conclusion(content)

        # Using GPT-3.5-turbo-instruct to generate content for each section
        intro_completion = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=intro,
            max_tokens=200,
        )

        specs_completion = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=specs,
            max_tokens=200,
        )

        conclusion_completion = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=conclusion,
            max_tokens=200,
        )

        blog_content = (
            intro_completion.choices[0].text.strip()
            + "\n\n"
            + specs_completion.choices[0].text.strip()
            + "\n\n"
            + conclusion_completion.choices[0].text.strip()
        )

        trending_urls = [article.url for article in self.trending_articles]


        blog_content+= self.find_similar_articles(trending_urls,num_results=3)


        return blog_content

    def generate_blog_post_batch(self):

        trending_ids = [article.id for article in self.trending_articles]
        print(trending_ids)

        response = metaphor.get_contents(trending_ids)
        total_content = ""

        for item in response.contents:
            cleaned_html = self.clean_html_and_get_text(f'''{item.extract}''')
            total_content+= cleaned_html
            total_content+= '\n******\n'

        post_content = self.generate_blog_post(total_content)

        print(post_content)
        return post_content



    def clean_html_and_get_text(self, html_input):
        # Removing HTML tags using BeautifulSoup
        soup = BeautifulSoup(html_input, 'html.parser')
        text_content = soup.get_text()

        return text_content

# Example usage:
product_name = "Apple watch"
category = "Smart Watch"  #Give your product category here
blog_post_generator = BlogPostGenerator(product_name, category)
amazon_blog_posts = blog_post_generator.generate_blog_post_batch()

