import openai
import os
import regex as re
import html

from dotenv import find_dotenv, load_dotenv
from newspaper import Article
from pygooglenews import GoogleNews
from google.cloud import translate_v2 as translate


'''
This module is designed to fetch the top trending news in the Google News' RSS
and analyze the sentiment of them.
'''
class News():

    def __init__(self):

        gn = GoogleNews(country='BR', lang='pt')
        self.top_news = gn.top_news()['entries']

        self.news = None

        print('classe news criada')


    def get_top_news(self, n=5):
        '''
        Gets the n top news from Google News RSS Brazil, in portuguese and
        organizes it in a dictionary.
        '''
        links = [self.top_news[i]['link'] for i in range(n)]

        for i, url in enumerate(links):
            article = Article(url, language='pt', fetch_images=False)
            article.download()
            article.parse()

            text = article.text
            text = text.replace('\n', ' ')
            text = text.strip()

            self.news[i] = {'url': article.url,
                            'title': article.title,
                            'text': text
                            }

    def translate_title(self):
        '''
        Receives the dict from get_news_from_google_feed, translates the title
        to english and adds a new 'title_translation' key to self.news.
        '''
        client = translate.Client()

        for content in self.news.values():
            title = content['title']
            trans_title = client.translate(title, source_language='pt-BR',
                                           target_language='en')['translatedText']
            trans_title = html.unescape(trans_title)
            content['title_translation'] = trans_title


    def translate_text(self, text):
        '''
        Receives a text in porguese and returns it translated to english.
        -----------
        Parameters: - text: A str with the text in pt-BR to be translated
        -----------
        Returns: - translation: The translated str text in english.
        '''
        client = translate.Client()

        translation = client.translate(text, source_language='pt-BR',
                                       target_language='en')['translatedText']
        translation = html.unescape(translation)

        return translation


    def get_text_resume(self, text):
        '''
        Receives an english text and returns it resumed and in pt-BR.
        -----------
        Parameters: - text: A str with the text in english to be resumed
        -----------
        Returns: - resume_translation: A str with the given text resumed and
                    translated to pt-BR.
        '''
        openai.api_key = os.getenv('OPENAI_API_KEY')

        translated_text = self.translate_text(text)

        prompt = 'Summarize this for a second-grade student:\n\n' \
            + translated_text + '\n'

        response = openai.Completion.create(
            engine='text-davinci-002',
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        resume = response['choices'][0]['text']

        client = translate.Client()

        resume_translation = client.translate(resume, source_language='en',
                                              target_language='pt-BR')['translatedText']
        resume_translation = html.unescape(resume_translation)

        return resume_translation


    def get_sentiment_of_news(self):
        '''
        Receives the dict with translated title from translate_title, sends the title
        to OpenAI API and returns the same dictionary with a new 'sentiment' key.
        '''
        path = find_dotenv()
        load_dotenv(path)
        openai.api_key = os.environ.get('OPENAI_API_KEY')

        phrases = [f'{content["title_translation"]}' for content in self.news.values()]
        prompt = 'Classify in positive, neutral or negative the sentiment in these phrases:\n\n'

        for i, phrase in enumerate(phrases):
            prompt += f'{i+1}. "{phrase}"\n'
        prompt += '\nPhrases sentiment ratings:\n'

        res = openai.Completion.create(
            engine='text-davinci-002',
            prompt=prompt,
            temperature=0,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        classification = res['choices'][0]['text'].strip().split('\n')
        clean_classification = [re.sub('\d. ', '', sentiment) \
            for sentiment in classification]

        for i, sentiment in enumerate(clean_classification):
            self.news[i]['sentiment'] = sentiment


    def get_news(self, n=5):
        '''
        Gets the top n news from Google News' RSS and returns a dictionary with
        the titles, text, translation of the title and sentiment of the article.
        ---------
        Receives:
            - n: The number of news to analyze
        ---------
        Returns: dictionary
            - self.news: {- n: {
                url: The article's URL
                title: The original title of the article
                text: The original text of the article
                translation: The translation of the title of the article
                sentiment: The classification of the sentiment of the article,
                           in negative, neutral or positive.
            }}
        '''
        self.news = {}
        self.get_top_news(n)
        self.translate_title()
        self.get_sentiment_of_news()

        for i, article in self.news.copy().items():
            if 'cnn' in article['title'].lower():
                _ = self.news.pop(i, None)

        return self.news


    def get_news_by_sentiment(self, n=5):
        '''
        Gets the dictionary with the news already analyzes and separates them
        into 3 lists, one for each possible sentiment.
        ----------
        Returns: tuple
            - positive_news: A list with the positive news.
            - neutral_news: A list with the neutral news.
            - negative_news: A list with the negative news.
        '''
        if not self.news:
            self.get_news(n)


        positive_news = []
        neutral_news = []
        negative_news = []
        for i, article in self.news.items():
            if article['sentiment'].lower() == 'positive'.lower():
                positive_news.append(self.news[i])
            elif article['sentiment'].lower() == 'neutral'.lower():
                neutral_news.append(self.news[i])
            else:
                negative_news.append(self.news[i])


        return positive_news, neutral_news, negative_news


if __name__ == '__main__':
    news = News()
    positive, neutral, negative = news.get_news_by_sentiment(n=7)

    traduzido = news.translate_text(neutral[0]['text'])
    resumo = news.get_text_resume(traduzido)
    print(resumo)
