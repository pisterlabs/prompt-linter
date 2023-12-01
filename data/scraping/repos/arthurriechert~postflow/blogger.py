from ghost.manager import GhostManager
from llm.openai.model import OpenAIModel

import llm.prompts as prompt
import settings as sg
from parse import parse_json

async def run_single_update (manager):
    """
    Runs a single loop of generating a blog post and publishing to ghost

    """
    
    # Initialize Ghost Admin API
    ghost = GhostManager()

    # Initialize OpenAI API
    llm = OpenAIModel()
    
    await manager.broadcast('<h2>Standby</h2><p style="margin-top: -15px">Postflow is gathering meta data to write your blog post</p>')

    # Get settings
    settings = sg.get_settings()

    # Organize settings
    description = sg.parse_settings(settings, "description")

    # Get a json containing topics
    topics = prompt.get_new_idea(description=description, current=ghost.titles, model=llm)

    await manager.broadcast('<h2>Researching</h2><p style="margin-top: -15px">Planning blog content</p>')

    # Get title
    title  = parse_json(topics, "title")[0]

    # Print diagnostics
    print(f"\n\033[32mRetrieved the following topics: {topics}\033[0m")

    # Get outline
    outline = prompt.get_post_outline(f"{topics}", llm)

    # Print diagnostics
    print(f"\n\033[32mRetrieved the following outline: {outline}\033[0m")

    await manager.broadcast('<h2>Writing</h2><p style="margin-top: -15px">Generating your article using GPT-4</p>')

    # Write blog post as string
    article = prompt.write_blog_post(outline, llm)

    # Print diagnostics
    print(f"\n033[32mRetrieved the following article: {article}\033[0m")

    # Format the blog post for uploading
    formatted_article = prompt.format_blog_post(article, llm, title)

    print(f"Formatted Article:\n{formatted_article}")

    # Print diagnostics
    print(f'\033[32mFORMATTED ARTICLE: {formatted_article}')

    await manager.broadcast('<h2>Publishing</h2><p style="margin-top: -15px">Publishing new article to Ghost</p>')

    # Post the article
    ghost.post_article(json=formatted_article)
