import pandas as pd
from bs4 import BeautifulSoup
import requests
import urllib, pymysql, calendar, time, json
from urllib.request import urlopen
from datetime import datetime
from threading import Timer
import lxml
import re
import os
import openai

openai.api_key = "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"


# 네이버 뉴스 데이터 DB 업데이트
class DBUpdater:
    def __init__(self):
        ''' 유의사항:
            해당 코드는 date가 primary key로 설정되지 않으면 date값을 여러개를 가져오지 못한다.  
            DB에서 직접 date를 기본키로 설정 해주는 것이 편하다.
        '''
        self.conn=pymysql.connect(host='3.38.94.77', user='root', password='1234',db='kosletter',charset='utf8mb4')
        
        with self.conn.cursor() as curs:
            sql="""
            CREATE TABLE IF NOT EXISTS companys_news(
                id BIGINT,
                code VARCHAR(20),
                title LONGTEXT,
                content LONGTEXT,
                summary LONGTEXT,
                link VARCHAR(200),
                PRIMARY KEY(ID)
            );
            """
            curs.execute(sql)
        self.conn.commit()

        self.codes = dict()
    
    def __del__(self):
        '''소멸자 정의'''
        self.conn.close()
        
    
    
    def crawl_main(self):
        '''네이버 증권 뉴스 크롤링 후 저장 (뉴스제목, 뉴스본문, 뉴스링크)'''
        
        def crawler(company_code, maxpage):
            
            page = 1 
            
            while page <= int(maxpage): 
            
                url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(page) 
                source_code = requests.get(url).text
                html = BeautifulSoup(source_code, "lxml")
            
        
                # 뉴스 제목 
                titles = html.select('.title')
                title_result=[]
                for title in titles: 
                    title = title.get_text() 
                    title = re.sub('\n','',title)
                    title_result.append(title)
        
        
                # 뉴스 링크
                links = html.select('.title') 
        
                link_result =[]
                for link in links: 
                    add = 'https://finance.naver.com' + link.find('a')['href']
                    link_result.append(add)
        
        
                # 뉴스 날짜 
                dates = html.select('.date') 
                date_result = [date.get_text() for date in dates] 
        
        
                # 뉴스 매체     
                sources = html.select('.info')
                source_result = [source.get_text() for source in sources] 
                
                # 뉴스 본문
                content_result = []
                for link in link_result:

                    # User-Agent 헤더를 설정하여 크롤링하는 것처럼 보이도록 함
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"
                    }

                    # 웹페이지에 GET 요청 보내기
                    response = requests.get(link, headers=headers)

                    # 응답 내용을 BeautifulSoup을 사용하여 파싱
                    soup = BeautifulSoup(response.content, "html.parser")

                    # 기사 본문 내용 추출
                    article_content = soup.find("div", class_="scr01")
                    if article_content:
                        # 본문 내용 출력
                        # print(article_content.get_text())
                        content_result.append(article_content.get_text())
                    else:
                        print("기사 내용을 찾을 수 없습니다.")
                        content_result.append("fail")
        
        
                # 변수들 합쳐서 해당 디렉토리에 csv파일로 저장하기 
        
                result= {"code" : [company_code]*len(title_result), "date" : date_result, "journal" : source_result, "title" : title_result, "link" : link_result, "content" : content_result} 
                df_result = pd.DataFrame(result)
                df_result5 = df_result.iloc[:5]
                
                print("다운 받고 있습니다------")                
        
                page += 1 
            
            return df_result5
        
        # 종목코드 추출
        data = pd.read_csv('C:/Users/cjs45/Desktop/Hackathon/hackathon3/data_5330_20230812.csv', encoding='cp949')
        
        result_df = pd.DataFrame()
        
        codes = data['종목코드'].map('{:06d}'.format)
        
        for code in codes:
            
            company = code
        
            # maxpage = input("최대 뉴스 페이지 수 입력: ")
    
            result_df = pd.concat([result_df, crawler(company, '1')])
            
        
        result_df.reset_index(inplace=True)
        result_df.reset_index(inplace=True)
        
        result_df = result_df[["index", "code", "title", "content", "link"]]
    
        return result_df
        
    
    def summary_chatgpt_api(self, content):
        '''ChatGPT 3.5 버전으로 뉴스 본문을 1문단으로 요약해 저장'''
        
        messages = []
        
        content = content + "\\ 위 내용을 한 문단으로 요약해줘"
        
        messages.append({"role":"user", "content":content})

        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages=messages
        )

        chat_response = completion.choices[0].message.content
        # print(f'ChatGPT: {chat_response}')
        messages.append({"role":"assistant", "content":chat_response})
        
        summary = chat_response
        
        return summary
    
    
        
    def update_comp_info(self):
        '''주식 시세를 companys_news 테이블에 업데이트한 후 딕셔너리 저장'''
        sql="SELECT * FROM companys_news"
        df = pd.read_sql(sql, self.conn)
        
        with self.conn.cursor() as curs:
            sql = "SELECT max(code) FROM companys_news"
            curs.execute(sql)
            rs = curs.fetchone()
            today = datetime.today().strftime('%Y-%m-%d')

            news=self.crawl_main()
            for idx in range(len(news)):
                newIdx = int(news.index.values[idx]) + 1
                code = news.code.values[idx]
                title = news.title.values[idx].replace("'", "''")
                content = news.content.values[idx].replace("'", "''")
                
                if idx < 25:
                    summary = self.summary_chatgpt_api(content)
                else:
                    summary = "None"
                
                link = news.link.values[idx]
                sql = f"REPLACE INTO companys_news (id, code, title, content, link, summary)"\
                    f"VALUES ('{newIdx}', '{code}','{title}','{content}', '{link}', '{summary}')"
                curs.execute(sql)
            self.conn.commit()
            print('')



    def execute_daily(self):
        '''실행 즉시 daily_price 테이블 업데이트'''
        self.update_comp_info()


if __name__=='__main__':
    dbu=DBUpdater()
    dbu.execute_daily()