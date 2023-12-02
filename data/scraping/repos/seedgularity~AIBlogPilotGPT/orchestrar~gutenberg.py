import sys
import openai
from dotenv import load_dotenv
import colorama
colorama.init(autoreset=True)
from config import Config
import re

from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

cfg = Config()

# Configure OpenAI API key
try:
    openai.api_key = cfg.openai_api_key
    browserless_api_key = cfg.browserless_api_key
    llm_model = cfg.llm_model
except KeyError:
    sys.stderr.write("OpenAI key configuration failed.")
    exit(1)


def convert_to_gutenberg_blocks(text):
    # Split the text into paragraphs
    paragraphs = text.split('\n\n')

    # Initialize an empty list to hold the blocks
    blocks = []

    # Iterate over each paragraph
    for paragraph in paragraphs:
        # Split the paragraph into lines
        lines = paragraph.split('\n')

        # Initialize an empty string to hold the current paragraph
        current_paragraph = ""

        # Iterate over each line
        for line in lines:
            # Check if the line is a headline
            if re.match(r"^<h[1-6]>", line):
                # If there's a current paragraph, add it as a block
                if current_paragraph:
                    blocks.append(create_paragraph_block(current_paragraph))
                    current_paragraph = ""

                # Add the headline as a block
                blocks.append(create_heading_block(line))

            # Check if the line is a link
            elif re.search(r"\[(.*?)\]\((.*?)\)", line):
                # If there's a current paragraph, add it as a block
                if current_paragraph:
                    blocks.append(create_paragraph_block(current_paragraph))
                    current_paragraph = ""

                # Add the link as a block
                blocks.append(create_link_block(line))

            # Otherwise, add the line to the current paragraph
            else:
                current_paragraph += line

        # If there's a current paragraph, add it as a block
        if current_paragraph:
            blocks.append(create_paragraph_block(current_paragraph))

    # Return the blocks as a HTML string
    return '\n'.join(blocks)

def create_paragraph_block(text):
    # Replace **text** with <strong>text</strong>
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    return '<!-- wp:paragraph --><p>{}</p><!-- /wp:paragraph -->'.format(text)

def create_link_block(text):
    # Replace [text](url) with <a href="url">text</a>
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)
    return '<!-- wp:paragraph --><p>{}</p><!-- /wp:paragraph -->'.format(text)

def create_heading_block(text):
    level = re.search(r"<h([1-6])>", text).group(1)
    text = re.sub(r"<\/?h[1-6]>", "", text)
    return '<!-- wp:heading {{"level": {}}} --><h{}>{}</h{}><!-- /wp:heading -->'.format(level, level, text.strip(), level)

def create_list_block(text):
    return '<!-- wp:list --><ul><li>{}</li></ul><!-- /wp:list -->'.format(text.replace("-", "").strip())

def create_image_block(text):
    alt_text, url = re.match(r"^!\[(.*)\]\((.*)\)", text).groups()
    return '<!-- wp:image {{"url": "{}", "alt": "{}"}} --><figure class="wp-block-image"><img src="{}" alt="{}"/></figure><!-- /wp:image -->'.format(url, alt_text, url, alt_text)

def create_quote_block(text):
    return '<!-- wp:quote --><blockquote>{}</blockquote><!-- /wp:quote -->'.format(text.replace("> ", "").strip())


##################### ONLY FOR TESTING #####################
def post_to_wordpress(title, content, category, tags):
    wp_url = cfg.site_url
    wp_username = cfg.wp_admin_username
    wp_password = cfg.wp_admin_password
    wp_blogid = ""

    wp = Client(wp_url, wp_username, wp_password)

    post = WordPressPost()
    post.title = title
    post.content = content
    post.terms_names = {
        'category': [category],
        'post_tag': tags
    }
    post.post_status = 'draft'  # Set the status of the new post.

    wp.call(NewPost(post))

def main():
    # Open the file in read mode ('r')
    with open('../How to Make Your Own Eco-Friendly Shopping Bags.txt', 'r') as file:
        # Read the file content and store it in the 'chunk' variable
        chunk = file.read()

    # Now you can use the 'chunk' variable in the convert_to_gutenberg_blocks function
    gutenberg_blocks = convert_to_gutenberg_blocks(chunk)
    post_to_wordpress("How to Make Your Own Eco-Friendly Shopping Bags", gutenberg_blocks, "Climate Change", ["Climate Change"])

if __name__ == "__main__":
    main()