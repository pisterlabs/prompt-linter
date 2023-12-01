from collections import Counter
import openai
import os
from dotenv import load_dotenv
import re
from NCHU_nlptoolkit.cut import *
import requests
from datetime import datetime, timedelta
import json


class Word_Fetcher:
    def __init__(self) -> None:
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.current_directory = os.getcwd()

    def check_folder(self):
        # 檢查 result 資料夾是否存在
        result_folder = 'result'
        if not os.path.exists(result_folder):
            # 如果不存在，則創建資料夾
            os.makedirs(result_folder)
            print(f'已創建 {result_folder} 資料夾')
        else:
            print(f'{result_folder} 資料夾已存在')

    def search_by_keyword(self, keyword: str, size: int = 100, page_from: int = 0, start: int = 0, end: int = 0) -> list:
        if start == 0 or end == 0:
            # 取得今天凌晨12點的時間
            today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
            today_timestamp = int(today.timestamp())
            # 取得昨天凌晨12點的時間
            yesterday = today - timedelta(days=1)
            yesterday_timestamp = int(yesterday.timestamp())
            start = yesterday_timestamp
            end = today_timestamp

        print(f'http://ptt-search.nlpnchu.org/api/GetByContent?size={size}&from={page_from}&start={start}&end={end}&content={keyword}')
        r = requests.get(f'http://ptt-search.nlpnchu.org/api/GetByContent?size={size}&from={page_from}&start={start}&end={end}&content={keyword}', headers={'Accept': 'application/json'})
        res = r.json()
        title_and_description_ptt = []
        record_title = [] # record article title to avoid same title duplicate record in result
        for result in res['hits']:
            # cleaning text
            text = result['_source']['article_title'].strip()
            content = result['_source']['content'].replace("\n"," ").strip()
            text = text.replace('\u3000', ' ') # \u3000是全形空白
            content = content.replace('\u3000', ' ')
            # 尋找 [任意字元] 對主題以及標題作切割
            match = re.search(r'\[(.+?)\]', text)
            if match:
                article_theme = match.group(1)  # 取得問卦，ie, []裡的內容
                result = re.sub(r'\[.+?\]', '', text)  # 刪除 [問卦] 及其中的文字
                article_title = result.strip()
                # check duplicate article_title
                if article_title not in record_title:
                    record_title.append(article_title)
                    title_and_description_ptt += cut_sentence(article_title, minword=2) + cut_sentence(content, minword=2)
                else:
                    title_and_description_ptt += cut_sentence(content, minword=2)
            else:
                print(text + content + ' not match')
                
        return title_and_description_ptt

    def generate_prompt(self, table: dict) -> str:
        # prompt = """以下文字的格式為 詞彙 。幫我根據這些詞彙分成三個分群，每個類別給我25~35個詞彙。\n輸出格式為：\n數字-此類別所代表的主題\n詞彙\n文字為以下\n###\n
        # """
        prompt = """請幫我將以下的詞彙分成三群，並針對每一群的主題進行說明。\n輸出格式為：\n數字-此群的主題說明文字:\n主題說明:<主題的說明>\n結果:<此群的詞彙用,隔開>\n###\n{"""
        for word, freq in table.items():
            # prompt += (word + ":" + str(freq) + "\n")
            prompt += (word + "\n")
        prompt += "}###"
        return prompt

    def chatGPT_clustering_words(self, table: dict):
        print("=============== Call ChatGPT ===============")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "user", "content": self.generate_prompt(table)}],
            temperature=0
            # frequency_penalty=0,
            # presence_penalty=0
        )
        return response['choices'][0]['message']['content']

    def save_response(self, search_keyword, text):
        print("=============== Save chatGPT response ===============")
        with open(f'{self.current_directory}/result/{search_keyword}.txt', 'w') as f:
            f.write(text)

    def get_top_K(self, frequency_counting_table, K) -> dict:
        if len(frequency_counting_table) < K:
            return dict(sorted(frequency_counting_table.items(), key=lambda x: x[1], reverse=1))
        else:
            return dict(sorted(frequency_counting_table.items(), key=lambda x: x[1], reverse=1)[0:K])

    def generate_dictionary_response(self, keyword, text) -> dict:
        print("=============== Generate Dictionary ===============")
        group_name_list = []
        groups = re.findall(r'\d-\w+', text)
        group_name_list.append(groups[0].split('-')[1])
        group_name_list.append(groups[1].split('-')[1])
        group_name_list.append(groups[2].split('-')[1])

        print(f"group_name_list is {group_name_list}")

        sections = re.split(r'\n\d-', text.strip())

        result_dict = {}
        i=0
        for section in sections[0:]:
            lines = section.strip().split('\n')
            content = lines[1].split(':')[1].strip()
            result_dict[group_name_list[i]] = {
                "主題說明": content,
                "結果": lines[2].split(':')[1].strip()
            }
            i+=1

        with open(f'{self.current_directory}/result/{keyword}.json', "w") as outfile:
            outfile.write(json.dumps(result_dict, indent=4, ensure_ascii=False))
        return result_dict
    

    """
    generate dictionary which format is 
    {
        "林書豪相關": {
            "主題說明": "林書豪是台灣籃球界的代表性人物，他的表現和成就引起了大家的關注和討論。",
            "結果": "林書豪, 台灣, 喜歡, 李雲翔, 本屆, 問題, 打球, 球季, 國家隊, 參賽, 中華隊, 一名, 中華籃協, 出賽, 當時, 工作, 新人, 面試, 公司, 興趣, 你很好, 轉一圈, 之後, 回來, 提到, 一定, 可帶, 知情, 沒給, 籃協, 中華奧會, 回覆, 不敵, 球迷, 質疑, 派出, 對此, 球員, 能夠, 透過, 輸給, 新世界, 排名, 公布, 位居, 第五, 樓下, 銅牌戰, 中國, 贏的, 機率, 冒出, 撥接, 林來瘋, 時期, 連勝, 場的, 對手, 都不, Nba, 討論, 且還, 全球華人, 籃球, 方面, 影響, 正面, 當年, 還在, 的時候, 說過, 大家都, 努力, 訓練, 體育館, 看見, 人有, 嚴重, 種族偏見"
        },
        "亞運相關": {
            "主題說明": "亞洲運動會（簡稱亞運）是亞洲最大的綜合性運動會，各國運動員在此競技，代表國家爭取榮譽。",
            "結果": "歸化, 亞運, 台灣, 阿提諾, 約旦, 國家, 奧會, 杭州亞運, Wbsc"
        },
        "籃球相關": {
            "主題說明": "籃球是一項全球性的運動，對於球迷和球員來說都有著重要的意義和影響。",
            "結果": "洋將, Kobe, 金身, 比較, Lb, 喬丹, Xd, ddd, 梗圖, 看見, 種族偏見"
        }
    }
    """
    def generate_dictionary(self, search_keyword, K):
        size = 100
        self.check_folder()
        # generate list about search_keyword
        title_and_description = self.search_by_keyword(search_keyword, size)
        # make frequency table about K
        frequency_counting_table = dict(Counter(title_and_description))
        top_K = self.get_top_K(frequency_counting_table, K)
        # Call chatGPT and save response
        res = self.chatGPT_clustering_words(top_K)
        self.save_response(search_keyword, res)
        # generate result json requirement's dictionary 
        dict_res = self.generate_dictionary_response(search_keyword, res)
        return dict_res
