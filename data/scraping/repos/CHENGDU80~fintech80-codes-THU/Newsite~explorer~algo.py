import requests
import json
from datetime import datetime, timedelta
import time
import csv
import pandas as pd
from newspaper import Article
from langchain.llms import OpenAI
import numpy as np
from sklearn import preprocessing             # 预处理数据
from scipy.stats.mstats import winsorize         # winsorize切数据
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ # 最重要的包，可以进行DFM


class Nowcasting():

    def __init__(self):
        self.res = None
        self.data = pd.read_excel('./explorer/gdp.xlsx')
        self.data.set_index('指标名称', inplace=True)


    def processing(self):
        data = self.data
        for each in data.columns:
            if each.endswith('当月同比'):
                data[each] = np.log(data[each] / 100 + 1)
            elif each.endswith('累计同比'):  # 累计同比需要换算成连续当月同比，即V1, 2V2-V1, 3V3-2V2,...,12V12-11V11
                temp = np.log(data[each] / 100 + 1)
                for i in range(len(temp)):
                    if i % 12:
                        temp[i] = temp[i] * ((i + 1) % 12) - (0 if pd.isna(temp[i - 1]) else temp[i - 1] * (i % 12))
                data[each] = temp
            elif each.endswith('当季同比'):
                data[each] = np.log(data[each] / 100 + 1)
            else:
                data[each] = np.log(data[each]).diff(12)
        data.replace([np.inf, -np.inf, 0], np.nan, inplace=True)  # 替换无穷为空
        data.dropna(axis=1, how='all', inplace=True)  # 丢弃全空列
        z_scaler = preprocessing.StandardScaler()
        z_data = z_scaler.fit_transform(data).T
        for i in range(len(z_data)):
            max_ = (z_data[i] > np.nanmean(z_data[i]) + np.nanstd(z_data[i]) * 3).sum() / len(z_data[i])
            min_ = (z_data[i] < np.nanmean(z_data[i]) - np.nanstd(z_data[i]) * 3).sum() / len(z_data[i])
            z_data[i] = winsorize(z_data[i], limits=[min_, max_])  # 这里估计为5%，缩一下即可
        data = pd.DataFrame(z_data.T, columns=data.columns, index=data.index)
        data_m = data[data.columns[1:]]  # 去除GDP的月频数据
        data_q = data[data.columns[:1]].dropna()  # GDP的季频数据
        data_m.index.freq = 'M'  # 月频
        data_q.index.freq = 'Q'  # 季频
        factors_settings = {'中国:GDP:不变价:当季同比': ['实际经济'],
                            '中国:工业增加值:当月同比': ['实际经济'],
                            '中国:工业企业:产销率:当月同比': ['实际经济'],
                            '中国:工业企业:出口交货值:当月同比': ['实际经济'],
                            '中国:工业企业:利润总额:当月同比': ['实际经济'],
                            '中国:CPI:当月同比': ['物价'],
                            '中国:CPI:不包括食品和能源(核心CPI):当月同比': ['物价'],
                            '中国:PPI:全部工业品:当月同比': ['物价'],
                            '中国:PPI:生活资料:当月同比': ['物价'],
                            '中国:PPIRM:当月同比': ['物价'],
                            '中国:CGPI:当月同比': ['物价'],
                            '中国:出口金额:当月同比': ['实际经济'],
                            '中国:进口金额:当月同比': ['实际经济'],
                            '中国:固定资产投资完成额:累计同比': ['实际经济'],
                            '中国:房地产开发投资完成额:累计同比': ['实际经济'],
                            '中国:民间固定资产投资完成额:累计同比': ['实际经济'],
                            '中国:固定资产投资本年新开工项目计划总投资额:累计同比': ['实际经济'],
                            '中国:固定资产投资本年施工项目计划总投资额:累计同比': ['实际经济'],
                            '中国:社会消费品零售总额:实际当月同比': ['实际经济'],
                            '中国:公共财政收入:当月同比': ['财政金融'],
                            '中国:公共财政支出:当月同比': ['财政金融'],
                            '31个大城市城镇调查失业率:当月同比': ['实际经济'],
                            '中国:制造业PMI': ['情绪'],
                            '中国:制造业PMI:新订单': ['情绪'],
                            '中国:制造业PMI:产成品库存': ['情绪'],
                            '中国:制造业PMI:从业人员': ['情绪'],
                            '中国:非制造业PMI:商务活动': ['情绪'],
                            '中国:非制造业PMI:新订单': ['情绪'],
                            '中国:非制造业PMI:从业人员': ['情绪'],
                            '中国:非制造业PMI:存货': ['情绪'],
                            '中国综合PMI:产出指数': ['情绪'],
                            '非官方中国PMI': ['情绪'],
                            '非官方中国服务业PMI:经营活动指数': ['情绪'],
                            '非官方中国综合PMI:产出指数': ['情绪'],
                            '中国:M2:当月同比': ['财政金融'],
                            '中国:官方储备资产(SDR口径)': ['财政金融'],
                            '中国:社会融资规模存量:当月同比': ['财政金融'],
                            '中国:社会融资规模存量:人民币贷款:当月同比': ['财政金融'],
                            '中国:全国政府性基金收入:累计同比': ['财政金融'],
                            '中国:全国政府性基金支出:累计同比': ['财政金融'],
                            '中国:产量:发电量:当月同比': ['实际经济'],
                            '中国:产量:汽车:当月同比': ['实际经济'],
                            '中国:货运量总计:当月同比': ['实际经济'],
                            '中国:货物周转量总计:当月同比': ['实际经济'],
                            '中国:70个大中城市新建商品住宅价格指数:当月同比': ['物价'],
                            '中国:70个大中城市二手住宅价格指数:当月同比': ['物价'],
                            '中国:用电量:工业:当月同比': ['实际经济'],
                            '中国:销量:汽车:当月同比': ['实际经济'],
                            }
        for factor in list(factors_settings.keys()):
            factors_settings[factor].append('增长')
        model = DynamicFactorMQ(data_m, factors=factors_settings, endog_quarterly=data_q)
        self.res = model.fit()  # 用最大似然和卡尔曼滤波

    def get_data(self):
        try:
            return pd.read_csv('./explorer/ready_now_cast.csv')
        except:
            if self.res is None:
                self.processing()
            states = self.res.states  # 这是迭代得到的各因子的隐状态
            return states.smoothed[['增长', '实际经济', '情绪', '物价', '财政金融']]


class Crawler():
    def __init__(self):
        self.llm = OpenAI(openai_api_key='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', model='text-davinci-003')

    def summary(self, text):
        # Prepare the text for summarization
        prompt = f"Provide a concise summary (under 100 words) regarding the following news article: {text}"
        response = self.llm.predict(prompt[:4000])
        return response

    def search_latest_news(self, time_before=0, limit=10, news="financial news"):
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": "1457c0390ec140139f32136e587836f4"
        }
        # Create an empty DataFrame with the desired columns
        df = pd.DataFrame(columns=["Published Date", "Title", "URL", "Content", "Summary"])

        # Load previously saved URLs to avoid duplicates
        seen_urls = set()
        try:
            with open("data.json", "r") as infile:
                previously_saved = json.load(infile)
                for item in previously_saved:
                    seen_urls.add(item['url'])
        except FileNotFoundError:
            pass  # If the file doesn't exist yet, just proceed

        sites = ['reuters.com', 'finance.yahoo.com', 'businessinsider.com', 'wsj.com', 'marketwatch.com',
                 'economist.com', 'cnbc.com']

        # Combine the list of sites into a single query string for the Bing API
        sites_query = ' OR '.join([f"site:{site}" for site in sites])

        def query_for_date(query_date):
            date_range = f"{query_date.strftime('%Y-%m-%d')}..{query_date.strftime('%Y-%m-%d')}"
            params = {
                "q": f"{news} AND ({sites_query})",  # Include the combined sites query here
                "responseFilter": "webpages",
                "freshness": date_range,
                "count": 200,
                "mkt": "en-US",
                "setLang": "en",  # Ensure that the results are in English
                "offset": 0
            }

            results_for_day = []

            while True:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'webPages' in data and 'value' in data['webPages']:
                        for result in data['webPages']['value']:
                            if result['url'] not in seen_urls:
                                results_for_day.append(result)
                                seen_urls.add(result['url'])
                        if len(data['webPages']['value']) < 10:
                            break
                        else:
                            params['offset'] += 10
                    else:
                        break
                else:
                    break
            return results_for_day

        all_results = []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_before)

        current_date = start_date

        while current_date <= end_date:
            print(f"Fetching articles on {current_date.strftime('%Y-%m-%d')}")
            daily_results = query_for_date(current_date)
            all_results.extend(daily_results)
            with open("data.json", "w") as outfile:
                json.dump(all_results, outfile)

            current_date += timedelta(days=1)
            time.sleep(0.1)

        # Read from a JSON file
        with open('data.json', 'r') as file:
            data = json.load(file)

        # Filter out entries with duplicate URLs
        seen_urls = set()
        to_remove = set()

        # Identify duplicate URLs
        for item in data:
            url = item['url']
            if url in seen_urls:
                to_remove.add(url)
            seen_urls.add(url)

        # Filter the data
        filtered_data = [item for item in data if item['url'] not in to_remove]

        # Store the filtered data back to the JSON file
        with open('data.json', 'w') as file:
            json.dump(filtered_data, file, indent=4)  # indent=4 for pretty printing

        print(f"Saved {len(all_results)} unique news articles to data.json.")

        with open("data.json", "r") as infile:
            articles = json.load(infile)

        # Write header to the CSV file first
        with open("data.csv", "w", newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Published Date", "Title", "URL", "Content", "Summary"])  # Header

        counter = 0

        # Create an empty list to store article data
        article_data = []

        for article in articles:

            if counter >= limit:
                break

            url = article['url']
            title = article['name']
            published_date = article.get('datePublished', 'N/A')

            # Fetch and parse the article's content
            news_article = Article(url)
            try:
                print(counter)
                news_article.download()
                news_article.parse()
                content = news_article.text.replace('\n', '')
                if content and len(content) > 100:
                    # Add data to the DataFrame
                    summ = self.summary(content).replace('\n', '')
                    # Add data to the list
                    article_data.append({
                        "Published Date": published_date,
                        "Title": title,
                        "URL": url,
                        "Content": content,
                        "Summary": summ
                    })
                    counter += 1
            except:
                pass
        df = pd.DataFrame(article_data, columns=["Published Date", "Title", "URL", "Content", "Summary"])

        return df

    def get_latest_news(self, time, limit, news="financial news", maximum=3):
        target_row_count = limit
        df = self.search_latest_news(time, limit, news)
        counter = 0
        while len(df) < target_row_count:
            if counter == maximum:
                break
            counter += 1
            new_df = self.search_latest_news(time, limit, news)
            df = pd.concat([df, new_df]).drop_duplicates()  # Concatenates and removes duplicates
        return df





