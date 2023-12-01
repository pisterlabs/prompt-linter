from __future__ import annotations
import requests
from lxml import etree
from io import BytesIO
import time
import pandas as pd
import numpy as np
import re
from datetime import datetime
import time
import sqlalchemy
import pyodbc
import base64
import urllib
import asyncio
from scipy.cluster.vq import kmeans, vq
import aiohttp
import sys
import psutil
from typing import Union, List, Callable
import pyarrow.parquet as pq
import concurrent.futures
from functools import partial
from openai.embeddings_utils import get_embedding, cosine_similarity
import h5py
import os
import pickle
import openai

API_KEY =
openai.api_key = API_KEY

class LbpamGpt:
    """ LBPAMGPT class is used to fetch news articles from paid & public datasources (bloomberg paid source is the only supported source at the moment)
        and run several AI powered operations over the data feed to create new management factors and trend detection tools. """
    
    
    def __init__(self):
        """ Initializing class variables. """
        self.active_df = pd.DataFrame()
        self.temp_df = pd.DataFrame()
        self.remaining_df = pd.DataFrame()
        self.keyword_df = pd.DataFrame()
        self.storage_dir = './h5_data_storage/'

    def save_as_pickle(self) -> LbpamGpt:
        """ Save LbpamGpt object as pickle. """
        with open('lbpamgpt_object_save.pickle', 'wb') as file:
            pickle.dump(self, file)
        return self



# ██████╗  █████╗ ████████╗ █████╗ ███████╗██████╗  █████╗ ███╗   ███╗███████╗    ███╗   ███╗ █████╗ ███╗   ██╗ █████╗  ██████╗ ███████╗███╗   ███╗███████╗███╗   ██╗████████╗
# ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝    ████╗ ████║██╔══██╗████╗  ██║██╔══██╗██╔════╝ ██╔════╝████╗ ████║██╔════╝████╗  ██║╚══██╔══╝
# ██║  ██║███████║   ██║   ███████║█████╗  ██████╔╝███████║██╔████╔██║█████╗      ██╔████╔██║███████║██╔██╗ ██║███████║██║  ███╗█████╗  ██╔████╔██║█████╗  ██╔██╗ ██║   ██║   
# ██║  ██║██╔══██║   ██║   ██╔══██║██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝      ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══██║██║   ██║██╔══╝  ██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   
# ██████╔╝██║  ██║   ██║   ██║  ██║██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗    ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║  ██║╚██████╔╝███████╗██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   
# ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝

    def load_df(self, df: Union[str, pd.DataFrame] = 'active_df_save.parquet', loading_shards: int = 100) -> LbpamGpt:
        """ Load a dataframe(df) as active dataframe. """
        if isinstance(df, str):
            try:
                parquet_file = pq.ParquetFile(df)
            except Exception as e:
                print(e)
                return
            num_rows = parquet_file.metadata.num_rows
            batch_size = num_rows // loading_shards
            df_chunks = []
            for i in parquet_file.iter_batches(batch_size=batch_size, use_pandas_metadata=True):
                df_chunks.append(i.to_pandas())
            self.active_df = pd.concat(df_chunks, axis=0)
        elif isinstance(df, pd.DataFrame):
            self.active_df = df
        else:
            print('error: df in load_df() should either be a .parquet filename or a pd.DataFrame() object.')
        return self

    def save_df(self, filename: str = 'active_df_save.parquet') -> LbpamGpt:
        """ Save dataframe in a .parquet file(filename). """
        if isinstance(filename, str):
            self.active_df.to_parquet(filename)
        else:
            print('error: filename in save_df() should be None or str type.')
        return self

    def split_df(self, column_name: str, percentage: float) -> LbpamGpt:
        """ Split the current self.active_df based on the provided column(column_name), keep the first (percentage)% as self.active_df and stores the other in self.remaining_df and saving it locally as remaining_df_save.parquet. """
        self.active_df = self.active_df.reset_index(drop=True)
        num_rows_per_ticker = self.active_df.groupby(column_name).size().mul(percentage).astype(int)

        sampled_df = self.active_df.groupby(column_name).apply(lambda x: x.sample(n=num_rows_per_ticker[x.name])).reset_index(drop=True)

        self.remaining_df = self.active_df[~self.active_df.index.isin(sampled_df.index)]
        self.remaining_df.to_parquet('remaining_df_save.parquet')
        self.active_df = sampled_df

        return self
    
    def set_temp_as_active(self) -> LbpamGpt:
        """ Set current self.temp_df as self.active_df. """
        self.active_df = self.temp_df.copy()
        return self
    
    def set_active_as_temp(self) -> LbpamGpt:
        """ Set current self.active_df as self.temp_df. """
        self.temp_df = self.active_df.copy()
        return self

    def concat_temp_with_active_df(self) -> LbpamGpt:
        """ Concat self.temp_df at the end of self.active_df. """
        self.active_df = pd.concat([self.active_df, self.temp_df], axis=0)
        return self

    def merge_requests_response(self, column_name: str = 'default') -> LbpamGpt:
        """ Merge results in self.storage_dir with current self.active_df as column(column_name). """
        files_to_read = os.listdir(self.storage_dir)
        response = []
        for pid in sorted(list(set([int(fname.split('_')[0]) for fname in files_to_read]))):
            for f in [file for file in files_to_read if str(pid) in file]:
                with h5py.File(self.storage_dir + f, 'r') as loaded_file:
                    fragment = loaded_file['data'][:].tolist()
                    response.append(fragment)
        response = [res.decode('utf-8') if isinstance(res, bytes) else res for subres in response for res in subres]
        self.active_df[column_name] = response
        return self

    def compute_clustering(self) -> LbpamGpt:
        """ Run a Kmean clustering algorithm over the provided column(column_name) from self.active_df. """
        data = np.array(self.keyword_df['embedding'].tolist())

        k = 1000

        centroids, distortion = kmeans(data, k)

        labels, _ = vq(data, centroids)

        self.keyword_df['cluster'] = labels
        self.active_df['cluster'] = self.active_df['keywords'].apply(lambda x: [self.keyword_df[self.keyword_df.keyword == keyword].cluster.iloc[0] for keyword in x])
        return
    



# ███████╗██╗  ██╗████████╗███████╗██████╗ ███╗   ██╗ █████╗ ██╗         ██████╗ ███████╗ ██████╗ ██╗   ██╗███████╗███████╗████████╗███████╗
# ██╔════╝╚██╗██╔╝╚══██╔══╝██╔════╝██╔══██╗████╗  ██║██╔══██╗██║         ██╔══██╗██╔════╝██╔═══██╗██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔════╝
# █████╗   ╚███╔╝    ██║   █████╗  ██████╔╝██╔██╗ ██║███████║██║         ██████╔╝█████╗  ██║   ██║██║   ██║█████╗  ███████╗   ██║   ███████╗
# ██╔══╝   ██╔██╗    ██║   ██╔══╝  ██╔══██╗██║╚██╗██║██╔══██║██║         ██╔══██╗██╔══╝  ██║▄▄ ██║██║   ██║██╔══╝  ╚════██║   ██║   ╚════██║
# ███████╗██╔╝ ██╗   ██║   ███████╗██║  ██║██║ ╚████║██║  ██║███████╗    ██║  ██║███████╗╚██████╔╝╚██████╔╝███████╗███████║   ██║   ███████║
# ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚══════╝   ╚═╝   ╚══════╝

    async def _request_keywords(self, str_element: str) -> openai.openai_object.OpenAIObject:
        """ Request to openai API technical keywords for a given piece of news(str_element). """
        str_element = str_element[:16000]
        prompt = "Extract 10 technical keywords from this text, meaningful to the text semantic:\n\n" + str_element + "\n\nI am interested in precise specific technical keywords excluding, company names, people names, and country names and any global non-specific terms."
        max_retries = 3
        retry_intervals = [5, 10, 15]  # Adjust the retry intervals as needed
        
        for retry_count in range(max_retries):
            try:
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=0.5,
                    max_tokens=200,
                    top_p=1.0,
                    frequency_penalty=0.8,
                    presence_penalty=0.0
                )
                if response is not None:
                    return response['choices'][0]['text']
            except Exception as e:
                pass
                
            if retry_count < max_retries - 1:
                sleep_interval = retry_intervals[retry_count]
                time.sleep(sleep_interval)
        
        return None

    async def _request_embedding(self, str_element: str) -> List[float]:
        """ Request to openai API the corresponding embedding for a given string element(str_element). """
        str_element = str_element[:35000]
        max_retries = 3
        retry_intervals = [5, 10, 15]  # Adjust the retry intervals as needed
        
        for retry_count in range(max_retries):
            try:
                embedding = get_embedding(str_element, engine="text-embedding-ada-002")
                if embedding is not None:
                    return embedding
            except Exception as e:
                pass
                
            if retry_count < max_retries - 1:
                sleep_interval = retry_intervals[retry_count]
                time.sleep(sleep_interval)
        
        return None

    def _launch_multiprocessing(self, func: Callable[str], column_name: str, subprocess_amount: int = 5) -> LbpamGpt:
        """ Launch multiprocessing on a given core numbers(subprocess_amount) over self.active_df specified column(column_name) through a given function(func). """
        self._update_storage()

        if psutil.cpu_count()/2 < subprocess_amount:
            subprocess_amount = psutil.cpu_count()/2

        chunk_size = len(self.active_df) // subprocess_amount
        
        chunks = [self.active_df[i:i + chunk_size] for i in range(0, len(self.active_df), chunk_size)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            for chunk in chunks:
                executor.submit(self._async_proxy, partial(self._launch_asynchronous_requests, func, column_name, chunk))
                time.sleep(2)
        return self

    async def _launch_asynchronous_requests(self, func: Callable[[str], None], column_name: str, df: pd.DataFrame, shard_amount: int = 30) -> None:
        """ Divide the provided pandas.Dataframe(df) into a given shard amount(shard_amount) to finally iterate over the specified column(column_name) of each shard sending requests using the provided request sending function(func). """
        if shard_amount > len(df):
            shard_amount = 1
    
        shard = len(df) // shard_amount
        pid = datetime.now().microsecond

        for x in range(0, len(df), shard):
            tasks = [func(str_elem) for str_elem in df[x:x+shard][column_name].tolist()]
            results = await asyncio.gather(*tasks)
            with h5py.File(f"{self.storage_dir}{pid}_{x}.h5", 'w') as f:
                f.create_dataset('data', data=results)

        return None

    def _async_proxy(self, async_partial: partial) -> LbpamGpt:
        """ Proxy function used to launch the provided asynchronous partial element(async_partial). """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(async_partial())
        return self

    def fetch_embeddings(self, column_name: str) -> LbpamGpt:
        """ Function used to fetch embedding for a self.active_df column. """
        self._launch_multiprocessing(self._request_embedding, column_name)
        self.merge_requests_response('embedding')
        return self

    def fetch_keywords(self, column_name: str, keywords_embedding: Bool = False) -> LbpamGpt:
        """ Function used to fetch keywords out of a self.active_df column(column_name). """
        self._launch_multiprocessing(self._request_keywords, column_name)
        self.merge_requests_response('keywords')
        self.active_df['keywords'] = self.active_df['keywords'].apply(lambda x: [keyword.lower() for keyword in self._extract_keywords(x)])
        if keywords_embedding:
            self.fetch_keywords_embeddings('keywords')
        return self

    def fetch_keywords_embeddings(self, column_name: str) -> LbpamGpt:
        """ Function used to fetch embeddings of keywords in a self.active_df column(column_name). """
        self.set_active_as_temp()
        self.active_df = pd.DataFrame([keyword for sublist in self.active_df[column_name] for keyword in sublist], columns=['keyword'])
        self.fetch_embeddings('keyword')
        self.keyword_df = self.active_df.copy()
        self.temp_df['keywords_embeddings'] = self.temp_df['keywords'].apply(lambda x: [self.active_df[self.active_df.keyword == word].embedding.iloc[0] for word in x])
        self.set_temp_as_active()
        return self



# ██╗███╗   ██╗████████╗███████╗██████╗ ███╗   ██╗ █████╗ ██╗         ██████╗ ███████╗ ██████╗ ██╗   ██╗███████╗███████╗████████╗███████╗
# ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗████╗  ██║██╔══██╗██║         ██╔══██╗██╔════╝██╔═══██╗██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔════╝
# ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝██╔██╗ ██║███████║██║         ██████╔╝█████╗  ██║   ██║██║   ██║█████╗  ███████╗   ██║   ███████╗
# ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██║╚██╗██║██╔══██║██║         ██╔══██╗██╔══╝  ██║▄▄ ██║██║   ██║██╔══╝  ╚════██║   ██║   ╚════██║
# ██║██║ ╚████║   ██║   ███████╗██║  ██║██║ ╚████║██║  ██║███████╗    ██║  ██║███████╗╚██████╔╝╚██████╔╝███████╗███████║   ██║   ███████║
# ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚══════╝   ╚═╝   ╚══════╝

    async def _fetch_news(self, session: aiohttp.ClientSession, ticker: str, start_date: str, custom_auth: str) -> Union[List[str], None]:
        """ Fetching story identifiers from db. """
        print('\r                                                 ', end='')
        print(f'\rfetching articles for {ticker}', end='')
        url = 'http://vm-srv63-mkl:9911/v1/resources/getNews?rs:ticker=' + ticker + f'&rs:class=34151&rs:startDate={start_date}'
        async with session.get(url, headers={'Authorization': 'Basic ' + custom_auth}) as response:
            result = await response.json()
            return result

    async def _fetch_story(self, session: aiohttp.ClientSession, story: str, custom_auth: str) -> Union[List[str], None]:
        """ Fetching story data from story identifier. """
        url = "http://vm-srv63-mkl:9911/v1/resources/stories?rs:suid=" + story
        async with session.get(url, headers={'Authorization': 'Basic ' + custom_auth}) as response:
            document = await response.read()
            return document

    async def _process_ticker(self, session: aiohttp.ClientSession, tickers: List[str], ticker: str, start_date: str, custom_auth: str) -> (Union[List[str], None], Union[List[str], None], Union[List[str], None], Union[List[tuple], None], Union[List[str], None]):
        """ Fetching and filtering news for a given ticker(ticker) from a given date(start_date). """
        response = await self._fetch_news(session, ticker, start_date, custom_auth)
        if isinstance(response['data'], list):
            identifiers = [x['StoryIdentifier'] for x in response['data']]
            print('\r                                                                               ', end='')
            print(f'\rnumber of articles for {ticker}: {len(identifiers)}', end='')
            headlines = []
            contents = []
            dates = []
            stocks_and_rates = []
            for story in identifiers:
                document = await self._fetch_story(session, story, custom_auth)
                tree = etree.parse(BytesIO(document))
                headline = tree.xpath("//Headline")[0].text
                body = tree.xpath("//Body")[0].text
                date = tree.xpath("//TimeOfArrival")[0].text
                ids_tickers = [ticker.text for ticker in tree.xpath("//AssignedTickers/ScoredEntity/Id")]
                scores_tickers = [ticker.text for ticker in tree.xpath("//AssignedTickers/ScoredEntity/Score")]
                stocks = list(zip(ids_tickers, scores_tickers))
                max_rate = max([int(x) for x in scores_tickers])
                # filtering max 4 diff compagnies in article + 1 of them is rates > 90 and is part of univers
                if len(ids_tickers) < 4:
                    for idx, score in enumerate(scores_tickers):
                        if int(score) > 90 and ids_tickers[idx] in tickers:
                            headlines.append(headline)
                            contents.append(body)
                            dates.append(date)
                            stocks_and_rates.append(stocks)
                            break
            return headlines, contents, dates, stocks_and_rates, ticker
        else:
            return None, None, None, None, ticker

    async def _fetch_routine(self, tickers: List[str], start_date: str, custom_auth: str):
        "News fetching routine used to gather news for a list of ticker(tickers) form a starting date(start_date)."
        HEADLINES_BATCH = []
        CONTENTS_BATCH = []
        DATES_BATCH = []
        STOCKS_AND_RATES_BATCH = []
        TICKERS_BATCH = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._process_ticker(session, tickers, ticker, start_date, custom_auth) for ticker in tickers]
            results = await asyncio.gather(*tasks)
            for headlines, contents, dates, stocks_and_rates, tick in results:
                HEADLINES_BATCH.append(headlines)
                CONTENTS_BATCH.append(contents)
                DATES_BATCH.append(dates)
                STOCKS_AND_RATES_BATCH.append(stocks_and_rates)
                TICKERS_BATCH.append(tick)
        return (HEADLINES_BATCH, CONTENTS_BATCH, DATES_BATCH, STOCKS_AND_RATES_BATCH, TICKERS_BATCH)

    async def fetch_articles(self, start_date: str, index_code: str) -> LbpamGpt:
        """ Asynchronous method to fetch articles from a given date(start_date) over a given univers (index_code), treating it and finally storing it in class variable accessible as temp_df. """
        print(f'starting article fetching with setup: {start_date} - {index_code}')

        auth = requests.auth.HTTPBasicAuth("admin", "admin")
        prod_server = 'http://vm-srv63-mkl:9911/v1/resources/getDocumentsForCategory'
        test_server = 'http://vm-srv60-mkl:9911/v1/resources/getDocumentsForCategory'
        credentials = f'admin:admin'
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        custom_auth = encoded_credentials
        serv_name_smartbeta = \
            """DRIVER={SQL Server};SERVER=sqlsmartbetaprod\\smartbetaprod;
                DATABASE=SMARTBETA_PROD;Trusted_Connection='Yes''"""
        smartbeta = pyodbc.connect(serv_name_smartbeta)
        # smartbeta server sqlalchemy connection
        quote_smartbeta = \
            urllib.parse.quote_plus(serv_name_smartbeta)
        sqlalch_conn = \
            r'mssql+pyodbc:///?odbc_connect={}'\
            .format(quote_smartbeta)
        engine = sqlalchemy.create_engine(sqlalch_conn)
        conn = smartbeta.cursor()

        query = f"""select t2.Bloom_Nego
        from
        (
        SELECT 
                                    distinct(  fsym_id)

                                from(
                                SELECT HCI.Index_Code,
                                HCI.Code_instrument,
                                HCI.date,
                                HCI.Weight_Pct as Weight_Pct,
                                RTRIM(IE.fsym_regional_id) as fsym_id
                                FROM [SMARTBETA_PROD].[dbo].[histo_comp_index] HCI
                                JOIN [SMARTBETA_PROD].[dbo].[instr_Equity] IE
                                ON HCI.Code_instrument = IE.Code_instrument
                                JOIN [SMARTBETA_PROD].[dbo].[company] C
                                ON C.fsym_security_id = IE.fsym_security_id
                                where HCI.Index_Code = '{index_code}'
                                AND HCI.date >= '{start_date}') A
                                left JOIN
                                (SELECT DISTINCT(fsym_id) as fsym2,
                                    start_date,
                                    end_date,
                                    Code_Cluster,
                                    RTRIM(value) as value
                                FROM [SMARTBETA_PROD].[dbo].[Style_Cluster_Data]
                                where Code_Cluster = 1) CLST
                                ON A.fsym_id = CLST.fsym2
                                WHERE A.date BETWEEN CLST.start_date
                                AND COALESCE(CLST.end_date, GETDATE())) T1
        join equity_info_codes() t2
        on t1.fsym_id = t2.fsym_regional_id"""

        compo = pd.read_sql_query(query, engine)

        compo = ['@'.join(x.split(' ')) for x in compo.Bloom_Nego.values.tolist()]

        tickers = [x.replace('@GY', '@GR').replace('@SQ', '@SM').replace('@SE', '@SW') for x in compo]

        t0 = time.time()

        HEADLINES_BATCH, CONTENTS_BATCH, DATES_BATCH, STOCKS_AND_RATES_BATCH, TICKERS_BATCH = await self._fetch_routine(tickers, start_date, custom_auth)

        print('\r                                                                               ', end='')
        print('\rfetching done!', end='')

        FILTERED_TICKERS, FILTERED_HEADLINES, FILTERED_CONTENTS, FILTERED_DATES, FILTERED_STOCKS_AND_RATES = zip(*[(x, y, z, w, o) for x, y, z, w, o in zip(TICKERS_BATCH, HEADLINES_BATCH, CONTENTS_BATCH, DATES_BATCH, STOCKS_AND_RATES_BATCH) if z is not None])

        df = pd.DataFrame({
            'ticker': FILTERED_TICKERS,
            'headline': FILTERED_HEADLINES,
            'content': FILTERED_CONTENTS,
            'date': FILTERED_DATES,
            'stocks_and_rates': FILTERED_STOCKS_AND_RATES
        })

        columns_to_explode = ['headline', 'content', 'date', 'stocks_and_rates']
        df_expanded = df.apply(lambda x: x.explode() if x.name in columns_to_explode else x)

        df_expanded = df_expanded.drop_duplicates(subset=['date', 'headline'])
        df_expanded = df_expanded[df_expanded.content.str.len() >= 300]
        df_expanded.date = pd.to_datetime(df_expanded.date)
        df_expanded['month'] = df_expanded.date.dt.month
        df_expanded['year'] = df_expanded.date.dt.year
        df_expanded['year_month'] = df_expanded.date.dt.strftime('%Y-%m')
        df_expanded['year_week'] = df_expanded.date.dt.strftime('%Y-%U')
        df_expanded['year_week'] = df_expanded.date.dt.strftime('%Y-%m-%d')

        print('\r                                                                               ', end='')
        print('\rtreating articles', end='')

        email_pattern = r'\n.*?[\w.+-]+@[\w-]+\.[\w.-]+\n'
        by_pattern = r'By\s[A-Za-z\s]+'
        bbg_pattern = r'\bBloomberg\b|\(Bloomberg\)|\[Bloomberg\]'
        special_characters_pattern = r'[^a-zA-Z0-9\s]'
        source_pattern = r'^To.*?\n'
        click_pattern = r'\n\s*To\b.*?\bhere(?:\n)?'
        def clear_trash(string):
            strr = re.sub(email_pattern, '\n', string)
            strr = re.sub(by_pattern, '', strr)
            strr = re.sub(bbg_pattern, '', strr, flags=re.IGNORECASE)
            strr = re.sub(special_characters_pattern, '', strr)
            strr = re.sub(source_pattern, '', strr, flags=re.MULTILINE)
            return re.sub(click_pattern, '', strr, flags=re.IGNORECASE | re.DOTALL)

        df_expanded.content = df_expanded.content.apply(lambda x: clear_trash(x))
        df_expanded = df_expanded[df_expanded.content.str.len() > 40]
        df_expanded.content = df_expanded.content.str.replace('\n', ' ')

        self.temp_df = df_expanded
        return self


# ██╗   ██╗████████╗██╗██╗     ███████╗
# ██║   ██║╚══██╔══╝██║██║     ██╔════╝
# ██║   ██║   ██║   ██║██║     ███████╗
# ██║   ██║   ██║   ██║██║     ╚════██║
# ╚██████╔╝   ██║   ██║███████╗███████║
#  ╚═════╝    ╚═╝   ╚═╝╚══════╝╚══════╝

    def _update_storage(self):
        """ Creating self.storage_dir directory if not already existing and removing all files present. """
        if self.storage_dir.split('/')[1] not in os.listdir():
            os.mkdir(self.storage_dir)
        files_in_storage = os.listdir(self.storage_dir)
        for file_to_remove in files_in_storage:
            os.remove(self.storage_dir + file_to_remove)

    def _extract_keywords(self, str_element: str) -> List:
        """ Extract keywords out of self.request_keywords output. """
        return re.findall(r'\b[A-Za-z]+\b', str_element)