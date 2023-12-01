import openai
import notion_api
import requests
import tenacity
import tiktoken
import base64
import configparser
import datetime
import os
import re
import arxiv
# 定义Reader类
OPENAI_API_KEYS = "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"
# the base URL for openai or other proxy
OPENAI_API_BASE = "https://api.openai.com/v1"
CHATGPT_MODEL = "gpt-3.5-turbo-0613"

class Reader:
    # 初始化方法，设置属性
    def __init__(self,
                 root_path='./',
                 ):


        self.language = 'English'


        self.root_path = root_path
        # 创建一个ConfigParser对象
        OPENAI_KEY = os.environ.get("OPENAI_KEY", "")
        # 获取某个键对应的值
        openai.api_base = OPENAI_API_BASE
        self.chat_api_list = [OPENAI_API_KEYS]
        self.chat_api_list.append(OPENAI_KEY)

        # prevent short strings from being incorrectly used as API keys.
        self.chat_api_list = [api.strip() for api in self.chat_api_list if len(api) > 20]
        self.chatgpt_model = CHATGPT_MODEL

        self.cur_api = 0
        self.file_format = 'md'
    
   
        self.gitee_key = ''
        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")


    def validateTitle(self, title):
        # 将论文的乱七八糟的路径格式修正
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        new_title = re.sub(rstr, "_", title)  # 替换为下划线
        return new_title

    def summary_with_chat(self, paper_list):
        htmls = []
        for paper_index, paper in enumerate(paper_list):
            # 第一步先用title，abs，和introduction进行总结。
            text = ''
            text += 'Title:' + paper.title
            text += 'Url:' + paper.url
            text += 'Abstract:' + paper.abs
            text += 'Paper_info:' + paper.section_text_dict['paper_info']
            # intro
            text += list(paper.section_text_dict.values())[0]
            chat_summary_text = ""
            try:
                chat_summary_text = self.chat_summary(text=text)
            except Exception as e:
                print("summary_error:", e)
                import sys
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                if "maximum context" in str(e):
                    current_tokens_index = str(e).find("your messages resulted in") + len(
                        "your messages resulted in") + 1
                    offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
                    summary_prompt_token = offset + 1000 + 150
                    chat_summary_text = self.chat_summary(text=text, summary_prompt_token=summary_prompt_token)

            htmls.append('## Paper:' + str(paper_index + 1))
            htmls.append('\n\n\n')
            htmls.append(chat_summary_text)
            notion_api.update_latest_page(chat_summary_text, col='Intro')

            # 第二步总结方法：
            # TODO，由于有些文章的方法章节名是算法名，所以简单的通过关键词来筛选，很难获取，后面需要用其他的方案去优化。
            method_key = ''
            methodology_list = ['Methodology', 'Research Design', 'Data Collection', 'Data Analysis', 'Experimental Procedures', 
                    'Survey Methods', 'Case Study Methods', 'Qualitative Methods', 'Quantitative Methods', 
                    'Sampling Methods', 'Statistical Analysis', 'Validity and Reliability', 'Ethical Considerations', 
                    'Instrumentation', 'Procedure','approach', 'experimental', 'experiment', 'experimentation', 'method', 'methodology', 'model', 'procedure', 'technique', 'algorithm', 'approach', 'method', 'methodology', 'model', 'procedure', 'technique', 'algorithm'
                    "Experiment and discussion", "Experiment and analysis", "Experiment and result", "Experiment and evaluation", "Experiment and comparison", "Experiment and performance", "Experiment and simulation", "Experiment and test", "Experiment and study", "Experiment and application", "Experiment and implementation", "Experiment and design", "Experiment and validation", "Experiment and verification", "Experiment and analysis", "Experiment and evaluation", "Experiment and comparison", "Experiment and performance", "Experiment and simulation", "Experiment and test", "Experiment and study", "Experiment and application", "Experiment and implementation", "Experiment and design", "Experiment and validation", "Experiment and verification", "Experiment and analysis", "Experiment and evaluation", "Experiment and comparison", "Experiment and performance", "Experiment and simulation", "Experiment and test", "Experiment and study", "Experiment and application", "Experiment and implementation", "Experiment and design", "Experiment and validation", "Experiment and verification"
                    ]
            # for parse_key in paper.section_text_dict.keys():
            #     if 'method' in parse_key.lower() or 'approach' in parse_key.lower():
            #         method_key = parse_key
            #         break
            keyword_pattern = re.compile(r'\b(' + '|'.join(methodology_list) + r')\b', re.IGNORECASE)

            # Iterate over each key in the dictionary
            for parse_key in paper.section_text_dict.keys():
                # Search for keywords using the regular expression pattern
                if keyword_pattern.search(parse_key):
                    method_key = parse_key
                    break

            if method_key != '':
                text = ''
                method_text = ''
                summary_text = ''
                summary_text += "<summary>" + chat_summary_text
                # methods                
                method_text += paper.section_text_dict[method_key]
                text = summary_text + "\n\n<Methods>:\n\n" + method_text
                chat_method_text = ""
                try:
                    chat_method_text = self.chat_method(text=text)
                except Exception as e:
                    print("method_error:", e)
                    import sys
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    if "maximum context" in str(e):
                        current_tokens_index = str(e).find("your messages resulted in") + len(
                            "your messages resulted in") + 1
                        offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
                        method_prompt_token = offset + 800 + 150
                        chat_method_text = self.chat_method(text=text, method_prompt_token=method_prompt_token)
                htmls.append(chat_method_text)
                notion_api.update_latest_page(chat_method_text, col='method')
            else:
                chat_method_text = ''
            htmls.append("\n" * 4)

            # 第三步总结全文，并打分：
            conclusion_key = ''
            for parse_key in paper.section_text_dict.keys():
                if 'conclu' in parse_key.lower():
                    conclusion_key = parse_key
                    break

            text = ''
            conclusion_text = ''
            summary_text = ''
            summary_text += "<summary>" + chat_summary_text + "\n <Method summary>:\n" + chat_method_text
            if conclusion_key != '':
                # conclusion                
                conclusion_text += paper.section_text_dict[conclusion_key]
                text = summary_text + "\n\n<Conclusion>:\n\n" + conclusion_text
            else:
                text = summary_text
            chat_conclusion_text = ""
            try:
                chat_conclusion_text = self.chat_conclusion(text=text)
            except Exception as e:
                print("conclusion_error:", e)
                import sys
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                if "maximum context" in str(e):
                    current_tokens_index = str(e).find("your messages resulted in") + len(
                        "your messages resulted in") + 1
                    offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
                    conclusion_prompt_token = offset + 800 + 150
                    chat_conclusion_text = self.chat_conclusion(text=text,
                                                                conclusion_prompt_token=conclusion_prompt_token)
            htmls.append(chat_conclusion_text)
            notion_api.update_latest_page(chat_conclusion_text, col='conc')
            htmls.append("\n" * 4)

            # # 整合成一个文件，打包保存下来。
            date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
            # export_path = os.path.join(self.root_path, 'export')
            # if not os.path.exists(export_path):
                # os.makedirs(export_path)
            # mode = 'w' if paper_index == 0 else 'a'
            # file_name = os.path.join(export_path,
                                    #  date_str + '-' + self.validateTitle(paper.title[:80]) + "." + self.file_format)
            # self.export_to_markdown("\n".join(htmls), file_name=file_name, mode=mode)
            # print("export_path: ", file_name)

            # NOTION API CALL
            # notion_api.update_latest_page("\n".join(htmls))
            htmls = []

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_conclusion(self, text, conclusion_prompt_token=800):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(len(text) * (self.max_token_num - conclusion_prompt_token) / text_token)
        clip_text = text[:clip_text_index]

        messages = [
            {"role": "system",
             "content": "You are a reviewer in the field of Computer Science and you need to critically review this article"},
            # chatgpt 角色
            {"role": "assistant",
             "content": "This is the <summary> and <conclusion> part of an English literature, where <summary> you have already summarized, but <conclusion> part, I need your help to summarize the following questions:" + clip_text},
            # 背景知识，可以参考OpenReview的审稿流程
            {"role": "user", "content": """                 
                 8. Make the following summary.Be sure to use {} answers (proper nouns need to be marked in English).
                    - (1):What is the significance of this piece of work?
                    - (2):Summarize the strengths and weaknesses of this article in three dimensions: innovation point, performance, and workload.                   
                    .......
                 Follow the format of the output later: 
                 8. Conclusion: \n\n
                    - (1):xxx;\n                     
                    - (2):Innovation point: xxx; Performance: xxx; Workload: xxx;\n                      
                 
                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.                 
                 """.format(self.language, self.language)},
        ]

        if openai.api_type == 'azure':
            response = openai.ChatCompletion.create(
                engine=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        # print("conclusion_result:\n", result)
        print("prompt_token_used:", response.usage.prompt_tokens,
              "completion_token_used:", response.usage.completion_tokens,
              "total_token_used:", response.usage.total_tokens)
        print("response_time:", response.response_ms / 1000.0, 's')
        return result

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_method(self, text, method_prompt_token=800):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(len(text) * (self.max_token_num - method_prompt_token) / text_token)
        clip_text = text[:clip_text_index]
        messages = [
            {"role": "system",
             "content": "You are a researcher in the field of Computer Science who is good at summarizing papers using concise statements"},
            # chatgpt 角色
            {"role": "assistant",
             "content": "This is the <summary> and <Method> part of an English document, where <summary> you have summarized, but the <Methods> part, I need your help to read and summarize the following questions." + clip_text},
            # 背景知识
            {"role": "user", "content": """                 
                 7. Describe in detail the methodological idea of this article. Be sure to use {} answers (proper nouns need to be marked in English). For example, its steps are.
                    - (1):...
                    - (2):...
                    - (3):...
                    - .......
                 Follow the format of the output that follows: 
                 7. Methods: \n\n
                    - (1):xxx;\n 
                    - (2):xxx;\n 
                    - (3):xxx;\n  
                    ....... \n\n     
                 8. One paragraph of summary (should start with 'in this paper'): \n\n
                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.                 
                 """.format(self.language, self.language)},
        ]
        if openai.api_type == 'azure':
            response = openai.ChatCompletion.create(
                engine=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        # print("method_result:\n", result)
        print("prompt_token_used:", response.usage.prompt_tokens,
              "completion_token_used:", response.usage.completion_tokens,
              "total_token_used:", response.usage.total_tokens)
        print("response_time:", response.response_ms / 1000.0, 's')
        return result

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_summary(self, text, summary_prompt_token=1100):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(len(text) * (self.max_token_num - summary_prompt_token) / text_token)
        clip_text = text[:clip_text_index]
        messages = [
            {"role": "system",
             "content": "You are a researcher in the field of Computer Science who is good at summarizing papers using concise statements"},
            {"role": "assistant",
             "content": "This is the title, author, link, abstract and introduction of an English document. I need your help to read and summarize the following questions: " + clip_text},
            {"role": "user", "content": """                 
                 2. list all the authors' names (use English)
                 3. mark the first author's affiliation (output {} translation only)                 
                 4. mark the keywords of this article (use English)
                 5. link to the paper, Github code link (if available, fill in Github:None if not)
                 6. summarize according to the following four points.Be sure to use {} answers (proper nouns need to be marked in English)
                    - (1):What is the research background of this article?
                    - (2):What are the past methods? What are the problems with them? Is the approach well motivated?
                    - (3):What is the research methodology proposed in this paper?
                    - (4):On what task and what performance is achieved by the methods in this paper? Can the performance support their goals?
                 Follow the format of the output that follows:                  
                 1. Title: xxx\n\n
                 2. Authors: xxx\n\n
                 3. Affiliation: xxx\n\n                 
                 4. Keywords: xxx\n\n   
                 5. Urls: xxx or xxx , xxx \n\n      
                 6. bibtex (generate the bibtex citations of this paper): xxx\n\n
                 7. Summary: \n\n
                    - (1):xxx;\n 
                    - (2):xxx;\n 
                    - (3):xxx;\n  
                    - (4):xxx.\n\n     
                 
                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed.                 
                 """.format(self.language, self.language, self.language)},
        ]

        if openai.api_type == 'azure':
            response = openai.ChatCompletion.create(
                engine=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.chatgpt_model,
                # prompt需要用英语替换，少占用token。
                messages=messages,
            )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        # print("summary_result:\n", result)
        print("prompt_token_used:", response.usage.prompt_tokens,
              "completion_token_used:", response.usage.completion_tokens,
              "total_token_used:", response.usage.total_tokens)
        print("response_time:", response.response_ms / 1000.0, 's')
        return result

    def export_to_markdown(self, text, file_name, mode='w'):
        # 使用markdown模块的convert方法，将文本转换为html格式
        # html = markdown.markdown(text)
        # 打开一个文件，以写入模式
        with open(file_name, mode, encoding="utf-8") as f:
            # 将html格式的内容写入文件
            f.write(text)

            # 定义一个方法，打印出读者信息

