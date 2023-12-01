import simple_salesforce
from html.parser import HTMLParser
from io import StringIO
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from lib.const import CONFIG_AUTHENTICATION, CONFIG_USERNAME, \
        CONFIG_PASSWORD, CONFIG_TOKEN
from lib.datasources.ds import Data, Content, Datasource
from lib.utils.lru import timed_lru_cache
from lib.model_manager import ModelManager


QUESTIONS_PROMPT = """Generate five questions that can be answered by the article with the summary:
    "{summary}"
    QUESTIONS:"""
SUMMARY_PROMPT = """Write a concise summary of the following:
    "{solution}"
    CONCISE SUMMARY:"""

class HtmlTagStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    stripper = HtmlTagStripper()
    stripper.feed(html)
    return stripper.get_data()

def get_authentication(auth_config):
    if CONFIG_USERNAME not in auth_config:
        raise ValueError(f'The auth config doesn\'t contain {CONFIG_USERNAME}')
    if CONFIG_PASSWORD not in auth_config:
        raise ValueError(f'The auth config doesn\'t contain {CONFIG_PASSWORD}')
    if CONFIG_TOKEN not in auth_config:
        raise ValueError(f'The auth config doesn\'t contain {CONFIG_TOKEN}')
    return {
            'username': auth_config[CONFIG_USERNAME],
            'password': auth_config[CONFIG_PASSWORD],
            'security_token': auth_config[CONFIG_TOKEN]
            }

class KnowledgeBaseSource(Datasource):
    def __init__(self, config):
        if CONFIG_AUTHENTICATION not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_AUTHENTICATION}')
        auth = get_authentication(config[CONFIG_AUTHENTICATION])
        self.sf = simple_salesforce.Salesforce(**auth)
        self.model_manager = ModelManager(config)

    def __generate_qeustions(self, summary):
        prompt = PromptTemplate.from_template(QUESTIONS_PROMPT)
        chain = (
                {'summary': RunnablePassthrough()}
                | prompt
                | self.model_manager.llm
                | StrOutputParser()
                )
        return chain.invode(summary)

    def __get_articles(self, start_date=None, end_date=None):
        clause = ''
        conditions = []
        if start_date is not None:
            conditions.append(f'LastModifiedDate >= {start_date.isoformat()}T00:00:00Z')
        if end_date is not None:
            conditions.append(f'LastModifiedDate < {end_date.isoformat()}T00:00:00Z')
        conditions.append(f'Knowledge_1_Approval_Status__c = \'Approval Complete\'')
        conditions.append(f'PublishStatus = \'Online\'')

        for condition in conditions:
            if clause:
                clause += ' AND '
            clause += condition
        sql_cmd = 'SELECT Id, KnowledgeArticleId, Title, Summary ' + \
                'FROM Knowledge__kav' + (f' WHERE {clause}' if clause else '')
        articles = self.sf.query_all(sql_cmd)

        for article in articles['records']:
            yield Data(
                    self.__generate_qeustions(article['Summary']),
                    {'article_id': article['KnowledgeArticleId'], 'title': article['Title']},
                    article['Id']
            )

    def get_initial_data(self, start_date):
        return self.__get_articles(start_date)

    def get_update_data(self, start_date, end_date):
        return self.__get_articles(start_date, end_date)

    @timed_lru_cache()
    def __get_summary(self, solution):
        prompt = PromptTemplate.from_template(SUMMARY_PROMPT)
        chain = (
                {'solution': RunnablePassthrough()}
                | prompt
                | self.model_manager.llm
                | StrOutputParser()
                )
        return chain.invoke(solution)

    def get_content(self, metadata):
        article = self.sf.query_all(f'SELECT Knowledge_1_Solution__c FROM Knowledge__kav '
                                    f'WHERE KnowledgeArticleId = \'{metadata["article_id"]}\'')
        return Content(
                {},
                self.__get_summary(strip_tags(article['records'][0]['Knowledge_1_Solution__c']))
                )

    def generate_output(self, content):
        return content.Summary
