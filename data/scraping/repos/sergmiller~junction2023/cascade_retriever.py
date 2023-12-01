import datetime
import os
import pandas as pd

import openai

from django.conf import settings

openai.api_key = settings.OPENAI_TOKEN

import tqdm



class CascadeRetriever:
    INSTANCE = None

    DATA_DIR = "ml"

    # here we control available information for model about world around, BASE_DATE simulate current date
    BASE_DATE = "2020-04-01"  # can be from 2018-03-18 to 2020-07-18

    def __init__(self):
        self.init_models()

    def init_models(self):
        import dsp
        # STOCK PRICES SOURCE
        self.load_prices()
        current_day_prices = self.upload_prices_info_up_to_date(CascadeRetriever.BASE_DATE, 30)
        self.prices_samples = [dsp.Example(question="ticker " + ticker, answer=make_sentence_for_ticker(ticker, current_day_prices[ticker])) for ticker
                          in tqdm.tqdm(self.tickers.keys(), position=0)]
        self.knn_prices = prepare_knn_source(dsp, self.prices_samples)

        print("Prepare prices source - OK")

        # NEWS SOURCE
        self.load_news()
        current_day_news = self.upload_news_up_to_date(CascadeRetriever.BASE_DATE, 25 * 60)
        self.news_samples = [dsp.Example(question=it, answer=None) for it in current_day_news]
        self.knn_news = prepare_knn_source(dsp, self.news_samples)

        print("Prepare news source - OK")


    def load_news(self):
        news_path = os.path.join(CascadeRetriever.DATA_DIR, "news_data/reuters_headlines.csv")
        self.news = pd.read_csv(news_path)
        from datetime import datetime

        def convert_date_format(input_date):
            # Convert the input string to a datetime object
            date_object = datetime.strptime(input_date, '%b %d %Y')

            # Convert the datetime object to the desired format
            formatted_date = date_object.strftime('%Y-%m-%d')

            return formatted_date

        self.news["Date"] = [convert_date_format(d) for d in self.news["Time"]]
        self.news["News"] = self.news["Description"]

    def upload_news_up_to_date(self, date, limit=None):
        res = self.news[self.news["Date"] <= date]["News"].values
        if limit is not None:
            res = res[-limit:]
        return res


    def load_prices(self):
        DIR = os.path.join(CascadeRetriever.DATA_DIR, "sp500_data")

        self.tickers = {}
        for t in tqdm.tqdm(os.listdir(DIR), position=0):
            if not t[-4:] == ".csv":
                continue
            self.tickers[t[:-4]] = pd.read_csv(os.path.join(DIR, t))[["Date", "Close"]]

    def upload_prices_info_up_to_date(self, last_date, history_len_days=7):
        prices = {}
        for i in range(history_len_days):
            for t in self.tickers.keys():
                seq = self.tickers[t]
                pos = seq[seq["Date"] >= last_date].index[0]
                prices[t] = self.tickers[t][pos - history_len_days + 1:pos + 1]
        return prices


    def process(self, message: str):
        import dsp
        import dspy

        lm = dspy.OpenAI(
            model='gpt-3.5-turbo')  # can be replaced with llama or another model, see https://github.com/stanfordnlp/dspy/blob/7d578638d070818f319dc892bb662c435d1cc1bd/docs/using_local_models.md#hfmodel
        # lm = dspy.HFClientTGI(model="meta-llama/Llama-2-7b-hf", port=8080, url="http://localhost")
        # lm = dspy.HFModel(model = 'meta-llama/Llama-2-7b-hf')

        # WIKI SOURCE(DEFINED AT APPLICATION TIME)
        colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
        dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

        print("Prepare wiki source - OK")

        passages = 5
        close_samples = [x["question"] for x in self.knn_news(dsp.Example(question=message, answer=None), passages)] \
            + [x["answer"] for x in self.knn_prices(dsp.Example(question=message, answer=None), passages)] \
            + dspy.Retrieve(k=passages)(message).passages
        print("SAMPLES: ", close_samples)
        chain = dspy.ChainOfThought("context, question -> answer")
        return chain(context=close_samples, question=message).answer


    @staticmethod
    def get_or_create():
        if CascadeRetriever.INSTANCE is None:
            CascadeRetriever.INSTANCE = CascadeRetriever()
        return CascadeRetriever.INSTANCE


def make_sentence_for_ticker(ticker, history):
    seq = "This is close prices for ticker " + ticker + " for last " + str(len(history))  + " days: "
    for l in history.values:
        date, vol = l
        seq += date + ":" + str(round(vol, 2)) + " "
    return seq


def prepare_knn_source(dsp, samples):
    with dsp.settings.context(vectorizer=dsp.SentenceTransformersVectorizer()):
        knn_func = dsp.knn(samples)
    return knn_func