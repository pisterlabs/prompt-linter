import sys
import json
import re
import requests
import lxml.html
import openai


def get_html(url):
    r = requests.get(url)
    html = lxml.html.fromstring(r.text)

    return lxml.html.tostring(html).decode('utf-8')

def get_article_text(html):
        article_html = lxml.html.fromstring(html)

        article_text = article_html.xpath('//p/text()')
        article_text = ''
        # for div in article_html.cssselect('div.paragraph'):
        #     article_text += ''.join(div.text_content())

        # return article_text
        for div in article_html.cssselect('div.paragraph'):
            yield div.text_content()


def get_article_summary(text):
    api_key = ''
    openai.api_key = api_key
    openai.organization = ''

    prompt = f'My fifth grader asked me what this passage means:\n\"\"\"\n{text}\n\"\"\"\nI rephrased it for him, in plain language a second grader can understand:\n\"\"\"\n'

    # tldr_prompt = f'{text}\n\ntl;dr:'
    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        max_tokens=140,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
        # stop=["\"\"\""]
        )

    return response



if __name__ == '__main__':
    url = sys.argv[1]
    html = get_html(url)
    article_text = get_article_text(html)
    # article_text ='During the sixties, while Lucille Ball was busy with her hit sitcom, "The Lucy Show," she also produced a daily radio program called "Let\'s Talk to Lucy." The program\'s tapes, which haven\'t been heard for decades, are now widely available via a SiriusXM channel and a new podcast.According to SiriusXM, "In between starring in four television series and more than 70 films, Lucy would jump from set to set—tape recorder in hand—to talk about all aspects of life with some of the biggest names in Hollywood history." The stars Ball chatted with were also her friends. There are 240 episodes in all, and in them Ball talks to stars of stage and screen including Carol Burnett, Frank Sinatra, Dean Martin, Mary Tyler Moore, Bing Crosby, Bob Hope, and Barbra Streisand.The radio show tapes, which have been living in the Ball family archives for decades, are now being repurposed as a podcast. Episodes have already begun to air on SiriusXM\'s pop-up channel, Channel 104, and they will soon be available to stream on other platforms, including Pandora, Stitcher, and the Sirius XM app.  Lucie Arnaz, Lucille Ball and Desi Arnaz\'s daughter, said in a press release, "Although I have been caretaking these ancient tapes for over 30 years, I had never really listened to them all and had no idea how many remarkable people Mom had talked to on these radio shows. It\'s a treasure trove of personal information from some of the greatest talents of American entertainment and my family and I can\'t wait to share them with the rest of the planet."In addition to the archival interviews, the podcast will also feature new programming that invites stars of today, including Ron Howard, Tiffany Haddish, Debra Messing, and Amy Poehler, to talk about Ball and her legacy.You can tune in and find more information about the podcast at siriusxm.com.What\'s your go-to podcast for daily listening? Have you heard any of the "Let\'s Talk to Lucy" programs?'

    for paragraph in article_text:
        text = paragraph + "\n tl;dr:"
        # print('Just downloaded the text: \n')
        # print(article_text)
        print()

        summary = get_article_summary(text)
        print(summary['choices'][0]['text'])