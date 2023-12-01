import os
import time
import requests
import google.auth
from google.oauth2.credentials import Credentials
from google.cloud import language_v1
import pandas as pd
from google.cloud import bigquery
import requests
from lxml import html
from ftfy import fix_text
from urllib.parse import urlparse
import openai


site_config = {
    'cheezburger.com': {
        'lists': {'title': '//h1[1]/text()', 'intro': '//*[@class="mu-description mu-theme-links"]'}
    },
    # 'cracked.com':'class="page-content"',
    'knowyourmeme.com':
    {'lists':
     {'title': '//div[@class="page-header"]/h2[1]/text()',
      'intro': '//div[@id="editorial-body"]/p[not(preceding-sibling::div[@class="collection-item"])]'},
      'guides':
     {'title': '//div[@class="page-header"]/h2[1]/text()',
      'intro': '//div[@id="editorial-body"]/p[not(preceding-sibling::div[@class="collection-item"])]'}
     }

}

models={
    'headline_kym_guides':'davinci:ft-literally-media:kym-guides-headline-generator-2023-03-08-14-57-47',
        'headline_chz_editorial':"davinci:ft-personal:intro-to-title-chz-training-ds-2-2023-03-06-11-12-19"
        } 


table_config = {'lists': 'headline_gai_list','guides':'headline_gai_guides',
                'scategories': 'sites_categories_1', 'entities': 'sites_entities_1'}
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/yanivbenatia/nlp_test/nlp.json'
OPEN_AI_KEY = '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'


class Fetcher:
    def __init__(self, site, top, offset=0, platform='discover', content_type='lists',model='headline_chz_editorial'):
        self.site = site
        self.top = top
        self.offset = offset
        self.platform = platform
        self.dataset_id = 'dbt_cdoyle'
        self.content_type = content_type
        self.table_id = table_config[content_type]
        self.model=models[model]

        self.training_dataset = []
        self.results_csv='{}_{}_{}_results.csv'.format(self.site, self.content_type,self.platform)
        self.training_csv='{}_{}_{}_training.csv'.format(self.site, self.content_type,self.platform)
        self.__init_bq__()

    def __init_bq__(self):
        # Set the credentials to use for the API calls
        self.credentials, self.project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"])

        # Set the client for the Natural Language API
        self.client = language_v1.LanguageServiceClient(
            credentials=self.credentials)

        # Set the client for the BigQuery API
        self.bq_client = bigquery.Client(
            project=self.project_id, credentials=self.credentials)

    def genarate_headline_from_intro(self, intro):
        intro += "\n\n###\n\n"
        openai.api_key = OPEN_AI_KEY
        response = openai.Completion.create(
            model=self.model,
            prompt=intro,
            temperature=0.8,
            max_tokens=350,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=[" END"])

        if "choices" in response:
            return response["choices"][0].text
        return 0

    def generate_query(self):
        if self.platform == 'discover':
            if self.site == 'knowyourmeme.com':
                query = f"""
                SELECT url, sum(clicks) as clicks
                FROM `literally-analytics.rivery.search_console_discover_by_url`
                WHERE property_id = 'sc-domain:{self.site}'
                and date > '2022-01-01' and url like '%/collections/%'
                group by 1
                ORDER BY 2 DESC
                LIMIT {self.top} offset {self.offset}
                """
            else:

                query = f"""
                SELECT url, sum(clicks) as clicks
                FROM `literally-analytics.rivery.search_console_discover_by_url`
                WHERE property_id = 'sc-domain:{self.site}'
                and date > '2023-01-01'
                group by 1
                ORDER BY 2 DESC
                LIMIT {self.top} offset {self.offset}
                """
        if self.platform == 'guides':
            query = f"""
                SELECT url, sum(clicks) as clicks
                FROM `literally-analytics.rivery.search_console_by_url`
                WHERE property_id = 'sc-domain:{self.site}'
                and date > '2022-01-01' and url like '%/guides/%'
                group by 1
                ORDER BY 2 DESC
                LIMIT {self.top} offset {self.offset}
                """
        if self.platform == 'facebook':
            query = f"""
            SELECT post_link_url_clean as url FROM `literally-analytics.production_facebook.fb_posts` 
            where post_type in ('instant_article_legacy','share') and   post_link_url_clean like '%{self.site}%'
            order by post_impressions desc 
            LIMIT {self.top} offset {self.offset}
            """
        if self.platform=='fb_reposts':
            query=f"""SELECT post_link_url_clean as url FROM `literally-analytics.production_facebook.fb_repost_generator` 
            where   post_link_url_clean like '%{self.site}%'
            and days_posted_group =90
            order by total_impressions desc 
             LIMIT {self.top} offset {self.offset}
            """
        return query

    def get_top_urls(self):
        query = self.generate_query()
        query_job = self.bq_client.query(query)
        urls = query_job.to_dataframe()
        return urls

    def generate_training_dataset(self):

        urls = self.get_top_urls()

        for index, row in urls.iterrows():
            url = row['url']
            page_title_and_intro = self.get_page_title_and_intro(url)
            if page_title_and_intro:
                self.training_dataset.append(page_title_and_intro)
            else:
                print('issue')
            time.sleep(0.3)

        df = pd.DataFrame(self.training_dataset, columns=[
                          "title", "intro", "url"])
        df.to_csv(self.training_csv)

        return

    def extract_content_from_html(self, r):
        try:
            # Get the text content from the response
            text = r.content.decode('utf-8')
            # Fix the text encoding using the ftfy library
            fixed_text = fix_text(text)

            # Parse the HTML content
            tree = html.fromstring(fixed_text)

            # Find the element with the class "page-content"
            path = site_config[self.site][self.content_type]
            # '//h1[1]/text()'
            page_headline = tree.xpath(path['title'])[0]
            page_content = tree.xpath(path['intro'])
            page_content = ' '.join(p.text_content() for p in page_content)

            response = (page_headline, page_content)

            return response
        except Exception:
            return False

    def make_request(self, url):
        try:
            # set header to mobile to get mobile layout
            headers = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1'}

            # Send a request to the URL
            r = requests.get(url, headers=headers)

            return r
        except Exception:
            return False

    def get_page_title_and_intro(self, url):
        try:
            r = self.make_request(url)
            response = self.extract_content_from_html(r)
            if response:
                response += (url,)
                return response
            return False
        except Exception:
            return False

    # load result into a BQ table

    def insert_result_to_db(self,filepath):
        if not filepath:
            filepath=self.results_csv
        print(
            f"Loading {self.site} {filepath} into {self.project_id}.{self.dataset_id}.{self.table_id}")

        # Convert the results to a dataframe
        
        df = pd.read_csv(filepath)

        # Set the client for the BigQuery API
        self.bq_client = bigquery.Client(
            project=self.project_id, credentials=self.credentials)
        # Get the table reference
        # Get the table reference
        table_ref = self.bq_client .dataset(
            self.dataset_id).table(self.table_id)

        # Try to get the table
        try:
            table = self.bq_client.get_table(table_ref)
        except:
            # If the table doesn't exist, set it to False
            table = False

        if table:
            # If the table exists, append the data to it
            self.bq_client .load_table_from_dataframe(df, table_ref).result()
        else:
            # If the table doesn't exist, create it and insert the data
            table = bigquery.Table(table_ref)
            self.bq_client.create_table(table)
            self.bq_client.load_table_from_dataframe(df, table_ref).result()

    def load_training_data(self,filepath=False):
        if not filepath:
            filepath=self.training_csv
        df = pd.read_csv(filepath)
        return df
    
    def run_generative_ai_on_headlines(self):
        output=[]
        df=self.load_training_data(filepath=self.training_csv)
        for index, row in df[0:200].iterrows():
            intro = row['intro']
            r=self.genarate_headline_from_intro(intro)
            output.append((row['url'],row['intro'],row['title'],r))

        df = pd.DataFrame(output, columns=["url","intro", "original_title","ai_title"])
        df.to_csv(self.results_csv)


        

