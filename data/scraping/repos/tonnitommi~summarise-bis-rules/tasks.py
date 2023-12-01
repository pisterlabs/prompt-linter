from robocorp.tasks import task
from robocorp import vault
from bs4 import BeautifulSoup
import requests
from RPA.HTTP import HTTP
from RPA.PDF import PDF
from RPA.Notifier import Notifier
from openai import OpenAI


@task
def summarize_new_things():

    # Set Robocorp libs up
    http = HTTP()
    pdf = PDF()

    openai_secret = vault.get_secret("OpenAI")

    client = OpenAI(
        api_key=openai_secret["key"]
    )

    links = get_links()
    baseurl = "https://www.bis.doc.gov"

    # only do the first for the funs
    dl_url = baseurl+links[0]
    print(dl_url)
    filename = "files/" + dl_url.split('=')[-1] + ".pdf"
    http.download(dl_url, filename)

    # then read the text
    text = pdf.get_text_from_pdf(filename)

    for page_number, content in text.items():
        rule_string = f'Page {page_number}:'
        rule_string = rule_string + content
        rule_string = rule_string + '\n---\n'

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant for the enterprise legal team, helping them to understand the newly updated Federal Buereau of Industry and Security rules and notifications.",
            },
            {
                "role": "user",
                "content": "Your task is to summarize the new rule or notification by the BIS, and highlight the parts that might be most relevant for global enterprise operations. Try avoiding to include the boilerplate language in your summary, but to focus directly on the actual relevant content. Aim for a summary that can be consumed by a legal person in less than a minute. Never drop relevant entity names, or enforcement dates from your summary. Always start with a one liner of what the rule or notice is about, followed by an empty line.\n\nBIS RULE CONTENT:\n" + rule_string,
            }
        ],
        model="gpt-4-1106-preview",
    )

    slack_it(completion.choices[0].message.content, dl_url)


def get_links():
    response = requests.get("https://www.bis.doc.gov/index.php/federal-register-notices")
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', text=lambda text: text and "BIS Rule" in text)
    return [link.get('href') for link in links]

def slack_it(message, link):
    slack_secrets = vault.get_secret("Slack")
    notif = Notifier()
    notif.notify_slack(
        message=f"NEW BIS NOTIFICATION SUMMARY:\n\n{message}\n\nLink: {link}",
        channel=slack_secrets["channel"],
        webhook_url=slack_secrets["webhook"]
    )
