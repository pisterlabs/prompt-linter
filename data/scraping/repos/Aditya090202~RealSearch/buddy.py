import openai
import os
import re
from dotenv import load_dotenv


# def summarize_user_advice(question, user_responses):

def summarize_user_advice(user_messages):
    load_dotenv()
    openai.api_key = os.getenv('openai_api_key')
    # Construct the prompt
    system_message = '''Given a question summarize answers from users, making your response short, concise and maintaining key ideas and topics. Make sure to keep response relevant to the query. Make sure to quote user. Always quote a user (@User) ALWAYS OR ELSE

Example input Query: "What is the best programming language to learn for future job opportunities?"

Output: Opinions vary, with some advocating for Python due to its growing use in various fields (@User1), others suggest JavaScript for its ubiquity in web development (@User2), while a few recommend focusing on Java or C# for enterprise environments (@User3)."'''
    # user_message = f"Question: \"{question}\"\n\n"
    # user_message += "\n".join([f"User:{user} \"{response}\"" for user, response in user_responses])

    # Make an API call to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_messages}
        ]
    )

    # Extract and return the summary from the API response
    return response['choices'][0]['message']['content']


def create_html_links(text, post):
    # Define a regular expression pattern to match usernames in the format (@username)
    pattern = r"@(\w+(?:-\w+)*)"
    global post_global
    post_global = post

    # Define a function that takes a match object and returns a string with the HTML link
    def replace_with_link(match):
        posts = post_global
        username = match.group(1)
        comment_url = ""
        found = False
        for post in range(len(posts)):
            for comment in posts[post][1]:
                if comment.author == username:
                    print("Same", comment.author, username)
                    submission = posts[post][0]
                    beg = f"https://www.reddit.com/r/{submission.subreddit.name}/comments/{submission.id}"
                    comment_url = beg + "/comment/" + comment.id
                    print(comment_url)
                    found = True
                    break
                else:
                    submission = posts[post][0]
                    comment_url = f"https://www.reddit.com/r/{submission.subreddit.name}/comments/{submission.id}"
            if found:
                break
        print(username)
        return f'<a href="{comment_url}" target="_blank">@{username if username is not "None" else "*User name unavailable*"}</a>'

    # Use re.sub() to replace all occurrences of the pattern with the HTML link
    html_output = re.sub(pattern, replace_with_link, text)
    return html_output


# Example usage:
question = "Looking for advice on backend to choose"
user_responses = [
    ("Fox", "You're more likely to build something maintainable in a language and framework you are familiar with."),
    ("420", "If you are comfortable with Python, I would highly recommend the Django web framework. Flask is another popular Python web framework."),
    ("drdro", "If you have already decided on Vue i'd suggest to go for Node.")
]
