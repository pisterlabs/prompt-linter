import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import re

class MailsSpider(CrawlSpider):
    name = 'mails'
    allowed_domains = ['gemengserv.com']
    start_urls = ['https://gemengserv.com/']

    rules = (
        Rule(LinkExtractor(allow=r''), callback='parse_item', follow=True),
    )
    
    def parse_item(self, response):
        
        html = response.text
        
        body = response.xpath('//*[contains(@class, "elementor-element-1785fac")]//text()').getall()
    
        shody = response.xpath('//*[@data-elementor-type="footer"]//text()').getall()
        for text in body:
            html = html.replace(text, '')
        for text in shody:
            html = html.replace(text, '')

        # Now, `html` contains the total response text without the elements matching the given xpath expressions
        
        email_found = False
        
        emails = re.findall(r'(?:Rebar|rebar)\s(?:Detailing|detailing)', html)
        for email in emails:
            if not email_found:
                yield {
                    'URL': response.url,
                    'Email': email
                }
                email_found = True
            else:
                break
            
# import pandas as pd

# from pandasai import PandasAI
# from pandasai.llm.openai import OpenAI

# df = pd.read_csv("t5.csv")

# df.head()

# # llm = OpenAI(api_token="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")
# # pandas_ai = PandasAI(llm, verbose=True, conversational=True)
# # response = pandas_ai.run(df, prompt="Delete Email column, then in URL column return urls as list items")


# # Delete Email column
# df = df.drop(columns=['Email'])

# # Return URLs as list items
# urls = df['URL'].tolist()

# print(urls)


 



