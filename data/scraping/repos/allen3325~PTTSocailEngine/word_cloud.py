from wordcloud import WordCloud
from collections import Counter
import openai
import os
from dotenv import load_dotenv
import re
from NCHU_nlptoolkit.cut import *
import requests
from datetime import datetime, timedelta


class Word_Cloud:
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

    def generate_picture(self, search_keyword, group_names: list):
        print("=============== Generate Wordcloud ===============")
        tmpdict = {}
        for i in [0, 1, 2]:
            tmpdict.clear()
            with open(f"{self.current_directory}/result/{search_keyword}-{group_names[i]}.txt") as f:
                data_list = f.readlines()
            f.close()
            for data in data_list:
                word = data.split(':')[1].strip()
                freq = data.split(':')[0].strip()
                if word != "" and freq != ":":
                    tmpdict[word] = int(freq)

            # generate wordcloud
            cloud = WordCloud(background_color="white", font_path=f"{self.current_directory}/fonts/POP.ttf", width=700,
                              height=350).generate_from_frequencies(tmpdict)
            cloud.to_file(f'{self.current_directory}/result/{search_keyword}-{group_names[i]}.png')

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

    def save_result(self,top_K , search_keyword, text) -> list:
        print("=============== Save chatGPT response to list ===============")
        group_name_list = []
        group_data_list = []
        groups = re.findall(r'\d-\w+', text)
        group_name_list.append(groups[0].split('-')[1])
        group_name_list.append(groups[1].split('-')[1])
        group_name_list.append(groups[2].split('-')[1])
        # split the group
        all_group = re.split(r'\d-\w+', text)
        # split word, frequency in group and append to the list
        for group in all_group:
            group = group.strip()
            group_data = ""
            # delete empty string
            if len(group) > 1:
                for word in group.split("\n"):
                    if word in top_K:
                        frequency = int(top_K[word])
                        group_data += f'{frequency}:{word.strip()}\n'
                    else:
                        print(f"top_K[{word}] is None")

                if group_data != "":
                    group_data_list.append(group_data)

        # save group list into file
        with open(f'{self.current_directory}/result/{search_keyword}-{group_name_list[0]}.txt', 'w') as f:
            f.write(f'{group_data_list[0]}')
        with open(f'{self.current_directory}/result/{search_keyword}-{group_name_list[1]}.txt', 'w') as f:
            f.write(f'{group_data_list[1]}')
        with open(f'{self.current_directory}/result/{search_keyword}-{group_name_list[2]}.txt', 'w') as f:
            f.write(f'{group_data_list[2]}')

        return group_name_list

    def save_response(self, search_keyword, text):
        print("=============== Save chatGPT response ===============")
        with open(f'{self.current_directory}/result/{search_keyword}.txt', 'w') as f:
            f.write(text)

    def get_top_K(self, frequency_counting_table, K) -> dict:
        if len(frequency_counting_table) < K:
            return dict(sorted(frequency_counting_table.items(), key=lambda x: x[1], reverse=1))
        else:
            return dict(sorted(frequency_counting_table.items(), key=lambda x: x[1], reverse=1)[0:K])

    def generate_word_cloud(self, search_keyword, K):
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
        group_name_list = self.save_result(top_K, search_keyword, res)
        # generate wordcloud
        self.generate_picture(search_keyword, group_name_list)
        return "Done."

    def test(self):
        print(f"{self.current_directory}/fonts/POP.ttf")
        print("/user_data/project/PTTSocailEngine/backend/fonts/POP.ttf")
        return "Test"
