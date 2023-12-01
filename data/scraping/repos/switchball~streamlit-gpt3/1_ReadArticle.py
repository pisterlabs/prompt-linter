# Read Article from WeChat

# Here is a Python code that can scrape articles from a WeChat public account and return the title and content of the article.

import streamlit as st
import requests
import math
import time
import io
import re
import pandas as pd
import openai
from bs4 import BeautifulSoup
from tempfile import NamedTemporaryFile
from utils.common_resource import get_tokenizer
from utils.remote_llm import RemoteLLM
from prompt import get_prompt_by_preset_id
from image import generate_article_image
from invite import InviteCodeCounter

MODEL_END_POINT = st.secrets.get("MODEL_END_POINT", "")

st.title("阅读小助手")
st.text("✨一键总结微信公众号文章，并给出阅读建议")


def get_invite_code_instance():
    if 'mysql' not in st.secrets:
        return None

    if 'icc' not in st.session_state:
        try:
            with st.spinner('启动中 ...'):
                st.session_state['icc'] = InviteCodeCounter(st.secrets["mysql"])
        except:
            st.write("网络故障. 3秒后自动重试")
            time.sleep(3)
            st.experimental_rerun()

    invite_code_counter = st.session_state['icc']
    # invite_code_counter = InviteCodeCounter(st.secrets["mysql"])

    url_code = st.experimental_get_query_params().get("code", None)

    if url_code is None or invite_code_counter.get_remain_times(url_code[0]) == -1:
        input_code = st.text_input("请输入邀请码", value="").upper()
        check_disabled = input_code == ""
        is_checking_code = st.button("确认", disabled=check_disabled)
        if is_checking_code:
            if invite_code_counter.get_remain_times(input_code) >= 0:
                st.success("success")
                st.balloons()
                with st.spinner("页面跳转中..."):
                    st.session_state['session_code'] = input_code
                    st.experimental_set_query_params(code=input_code)
                    time.sleep(2)
                    st.experimental_rerun()
            else:
                st.warning("该功能正在内测中 ... 邀请码不正确 ...")
                st.stop()
        else:
            st.stop()
    else:
        # valid code found
        st.session_state['session_code'] = url_code[0].upper()
    return invite_code_counter


invite_code_counter = get_invite_code_instance()


class Article:
    def __init__(self, url) -> None:
        self.headers = {
            'authority': 'mp.weixin.qq.com',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'cache-control': 'max-age=0',
            'cookie': 'pgv_pvid=1671376372258898; pgv_info=ssid=s1676372252078899; rewardsn=; wxtokenkey=777; wwapp.vid=; wwapp.cst=; wwapp.deviceid=; pac_uid=0_c4389ea7df8a1; ua_id=gJg4qg8cQpPkI9EfAAAAAIsLmLsCl4-U1_dIZ_VSF9M=; wxuin=76776862684849; uuid=2d4bf4f402427e3d1420f31c4d34852f; xid=506520fe2d89a0dfebec1da6fa33858c; mm_lang=zh_CN; ptui_loginuin=348988792; RK=dUWpQ/vZe8; ptcz=af8c4396797d39d05f3f87cf61e40f6eeb4186c4882654ede0e74c3e267d0932; skey=@B6Wq9TTyU; uin=o348988792; wedrive_uin=1688850523224148; wedrive_sid=AFRrWAD4U1YGg05CACc2YwAA; wedrive_sids=1688850523224148&AFRrWAD4U1YGg05CACc2YwAA; wedrive_skey=1688850523224148&c801ed3af1acc734f97529cf7b383368; wedrive_ticket=1688850523224148&CAESIMBDOqQ03uWl0BE7Ch4r5qssvqj1wsEO-AJiHjceff_R; xm_disk_vid=1688850523224148; xm_disk_corp_id=1970325010981265',
            'if-modified-since': 'Fri, 17 Mar 2023 23:47:25 +0800',
            'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.50',
        }
        self.title, self.content_df = self._get_wechat_article(url)

    def _get_wechat_article(self, url):
        """ Parse url and get wechat article title and content """
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content.decode("utf8"), 'html.parser')
        title = soup.find('h1', {'class': 'rich_media_title'}).text.strip()
        content = soup.find('div', {'class': 'rich_media_content'})

        stack = []
        stack = traverse2(content, level=0, stack=stack)
        content_df = pd.DataFrame.from_records(stack, columns=['name', 'level', 'font', 'text'])
        # content_text = "\n".join(x["text"] for x in stack if x["text"])
        return title, content_df
    
    def create_embedding(self, embedding_func, callback_func=None):
        """ Create embbeding for each line in content_df """
        self.content_df['embedding'] = None
        for i, row in self.content_df.iterrows():
            embedding = embedding_func(row['text'])
            row['embedding'] = embedding
            if callback_func:
                callback_func(i=i, total=len(self.content_df))
        return self.content_df

    def _cacl_article_avg_embedding(self):
        """ Calculate average embedding for the article """
        # some elements are weighted.
        pass

# 遍历 content 内部的所有标签，输出每个标签的内容
def traverse(element, level=0, file=None, suffix=""):
    # Set suffix = "\n" to save one paragraph to one line
    # NOTE: auto detect is more helpful
    for child in element.children:
        if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            file.write("\n" + "#" * int(child.name[1:]) + " " + child.get_text() + " " + suffix)
        elif child.name in ('p', 'span') and child.get_text().strip():
            file.write(child.get_text() + suffix)
        elif child.name == 'blockquote':
            file.write("> " + child.get_text() + suffix)
        elif child.name in ('section', 'body'):
            traverse(child, level+1, file)


def traverse2(element, level=0, stack=None, suffix="\n"):
    for child in element.children:
        if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            stack.append({'name': child.name, 'level': level, "text": child.get_text().strip()})
        elif child.name in ('p','span') and child.get_text().strip():
            # find font-size in style by regex
            patt = re.compile("font-size:\s*(\d+)")
            if 'font-size' in child['style']:
                match = patt.search(child['style'])
                font_size = int(match.group(1)) if match else None
            else:
                font_size = None
            stack.append({'name': child.name, 'level': level, "font": font_size, "text": child.get_text().strip()})
        elif child.name == 'blockquote':
            stack.append({'name': child.name, 'level': level + 1, "text": child.get_text().strip()})
        elif child.name in ('section', 'body'):
            stack.append({'name': child.name + '<begin>', 'level': level, 'text': ''})
            traverse2(child, level+1, stack, suffix)
            stack.append({'name': child.name + '<end>', 'level': level, 'text': ''})
    return stack

@st.cache_data(ttl=300)
def get_wechat_article(url, mode="simple"):
    headers = {
        'authority': 'mp.weixin.qq.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'cache-control': 'max-age=0',
        'cookie': 'pgv_pvid=1671376372258898; pgv_info=ssid=s1676372252078899; rewardsn=; wxtokenkey=777; wwapp.vid=; wwapp.cst=; wwapp.deviceid=; pac_uid=0_c4389ea7df8a1; ua_id=gJg4qg8cQpPkI9EfAAAAAIsLmLsCl4-U1_dIZ_VSF9M=; wxuin=76776862684849; uuid=2d4bf4f402427e3d1420f31c4d34852f; xid=506520fe2d89a0dfebec1da6fa33858c; mm_lang=zh_CN; ptui_loginuin=348988792; RK=dUWpQ/vZe8; ptcz=af8c4396797d39d05f3f87cf61e40f6eeb4186c4882654ede0e74c3e267d0932; skey=@B6Wq9TTyU; uin=o348988792; wedrive_uin=1688850523224148; wedrive_sid=AFRrWAD4U1YGg05CACc2YwAA; wedrive_sids=1688850523224148&AFRrWAD4U1YGg05CACc2YwAA; wedrive_skey=1688850523224148&c801ed3af1acc734f97529cf7b383368; wedrive_ticket=1688850523224148&CAESIMBDOqQ03uWl0BE7Ch4r5qssvqj1wsEO-AJiHjceff_R; xm_disk_vid=1688850523224148; xm_disk_corp_id=1970325010981265',
        'if-modified-since': 'Fri, 17 Mar 2023 23:47:25 +0800',
        'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.50',
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content.decode("utf8"), 'html.parser')
    if soup.find('h1', {'class': 'rich_media_title'}) is None:
        if soup.find('h2', {'class': 'weui-msg__title'}) is not None:
            err_msg = soup.find('h2', {'class': 'weui-msg__title'}).text.strip()
            st.warning('出错啦：' + err_msg)
        else:
            st.warning('服务器开小差啦')
        return None, None
    title = soup.find('h1', {'class': 'rich_media_title'}).text.strip()
    content = soup.find('div', {'class': 'rich_media_content'})
    # author = soup.find('span', {'id': 'profileBT'}).text.strip()
    # author = soup.find('span', class_='weui-media-box__meta').text.strip()  # 作者
    account_name = soup.find('strong', class_='profile_nickname').text.strip()  # 公众号名字
    if mode == 'simple':
        content_text = content.text.strip()
    else:
        with NamedTemporaryFile(mode="w+", delete=True, encoding="utf-8") as temp_file:
            traverse(content, file=temp_file)
            temp_file.seek(0)
            print(temp_file.name)
            content_text = temp_file.read()
            temp_file.close()
        # stack = []
        # stack = traverse2(content, level=0, stack=stack)
        # df = pd.DataFrame.from_records(stack, columns=['name', 'level', 'font', 'text'])
        # st.table(df)
        # content_text = "\n".join(x["text"] for x in stack if x["text"])
    return title, content_text, account_name


def test(text, temperature):
    result = RemoteLLM(MODEL_END_POINT).completion(input_text=text, temperature=temperature)
    st.json(result)
    msg = result["msg"]
    total_tokens = result["usage"]["total_tokens"]
    return msg, total_tokens


def paragraph_summary(text, temperature=0.1):
    message_list = [
        {"role": "system", "content": ""},
        # {"role": "user", "content": f"TL;dr 总结这段话：{text} 上文总结："},
        {"role": "user", "content": f"Instruction1=文段内容摘要 Instruction2 =推荐文段阅读理由 content={text} [abstract, recommendation]="},
        # {"role": "user", "content": f"TL;dr 总结这段话并写一个文章推荐理由：{text} 总结+推荐理由："},
        # {"role": "user", "content": f"Instructions: 以详略得当的方式总结文本内容 Content: {text} Summary:"},
    ]
    result = chat_completion(message_list, temperature=temperature, stream=True)
    # st.json(result)
    msg = result["choices"][0]['message']['content']
    total_tokens = result["usage"]["total_tokens"]
    return msg, total_tokens


def aggregate_summary(summary_list: list, temperature=0.1):
    if len(summary_list) == 1:
        return summary_list[0], 0
    num = len(summary_list)

    system_prompt = f"Description=An article is seperated into {num} parts equally, each of which has been summarized into abstract and recommended readings as follows. "
    user_prompt = f"Instruction=Extract the abstract and recommended readings from each part, and aggregate them to form a complete article summary and reasons for recommended readings.\n"
    for idx, summary in enumerate(summary_list):
        user_prompt += f'\n[abstract[{idx}], recommendation[{idx}]] = ```{summary}```'
    user_prompt += '\noutput_language=zh-cn'
    user_prompt += '\nconcluded [abstract, recommend reason] Properly detailed='

    message_list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        # {"role": "user", "content": f"Context Source=一些文段内容的总结 Instruction=聚合文段摘要，并给出推荐阅读理由 content={summary} [abstract, recommendation]="},
        # {"role": "user", "content": f"TL;dr 总结这段话并写一个文章推荐理由：{text} 总结+推荐理由："},
        # {"role": "user", "content": f"Instructions: 以详略得当的方式总结文本内容 Content: {text} Summary:"},
    ]
    result = chat_completion(message_list, temperature=temperature, stream=True)
    # st.json(result)
    msg = result["choices"][0]['message']['content']
    total_tokens = result["usage"]["total_tokens"]
    return msg, total_tokens


def generate_article_category(article_summary, temperature=0.6):
    message_list = [
        {"role": "system", "content": get_prompt_by_preset_id("给文章打标签 (AI)")},
        {"role": "user", "content": f"{article_summary}"},
    ]
    result = chat_completion(message_list, temperature=temperature, stream=False)
    msg = result["choices"][0]['message']['content']
    df = pd.read_table(io.StringIO(msg), sep='|')
    df.columns = [x.strip() for x in df.columns]
    category_list = []
    tag_col = '标签名' if '标签名' in df.columns else '标签' if '标签' in df.columns else None
    if tag_col is None:
        return []
    for idx, line in df.iterrows():
        tag = line[tag_col]
        # emoji = line['Emoji符号']
        if '---' in tag:
            continue
        category_list.append(tag)
    return category_list


def generate_title_emoji(article_title, temperature=0.6):
    message_list = [
        {"role": "system", "content": get_prompt_by_preset_id("标题转emoji (Fun)")},
        {"role": "user", "content": f"{article_title}"},
    ]
    result = chat_completion(message_list, temperature=temperature, stream=False)
    text = result["choices"][0]['message']['content']

    # return text
    # 定义正则表达式，匹配所有Emoji字符
    emoji_pattern = re.compile("["
        u"\U0001F000-\U0001FAFF"  # 1号平面
        u"\U00002600-\U000027FF"  # 0号平面
                           "]+", flags=re.UNICODE)

    # 使用正则表达式提取所有Emoji字符
    emojis = emoji_pattern.findall(text)

    # 将所有Emoji字符拼接成一个字符串
    result = "".join(emojis)
    return result

@st.cache_data(ttl=3600*6)
def chat_completion(
    message_list,
    model="gpt-3.5-turbo",
    temperature=0.6,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stream=False
    ):
    """ Chat completion """
    response = openai.ChatCompletion.create(
        model=model, messages=message_list, temperature=temperature, max_tokens=max_tokens, top_p=top_p, 
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, stream=stream
    )
    if stream:
        reply_msg = ""
        finish_reason = ""

        # streaming chat with editable slot
        reply_edit_slot = st.empty()
        for chunk in response:
            c = chunk['choices'][0]
            delta = c.get('delta', {}).get('content', '')
            finish_reason = c.get('finish_reason', '')
            reply_msg += delta
            reply_edit_slot.markdown(reply_msg)
        reply_edit_slot.markdown("")

        # calculate message tokens
        txt = "".join(m["content"] for m in message_list)
        input_tokens = len(get_tokenizer().tokenize(txt))
        completion_tokens = len(get_tokenizer().tokenize(reply_msg))

        # mock response
        response = {
            'choices': [{
                'message': {
                    'content': reply_msg,
                    'role': 'assistant'
                },
                'finish_reason': finish_reason
            }],
            'usage': {
                'total_tokens': input_tokens + completion_tokens
            }
        }
        return response
    else:
        return response

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    t = st.sidebar.number_input("temperature", 0.0, 1.0, value=0.2, step=0.1)
    url = st.text_input("请在下方粘贴文章地址:")
    if url.startswith("https://mp.weixin.qq.com"):
        debug_mode = st.sidebar.checkbox("调试模式", value=False, key="debug_mode")
        title, content, author = get_wechat_article(url, mode="simple")
        if title is None or content is None:
            st.stop()
        st.markdown("已获取文章标题：" + title)
        progress_text = "正在解析内容中... 请稍后"
        my_bar = st.progress(0.1, text=progress_text)
        tic = time.time()
        total_tokens = 0
        summary = ""
        content = content.replace('\t', ' ').replace('\n', ' ').replace('   ', ' ')
        # st.text_area("content", content)
        # st.write(content.split('\n'))
        for ctc in [content]:
            
            if len(ctc) >= 3000:
                chunk_num = math.ceil(len(ctc) / 3000)
                chunk_size = math.ceil(len(ctc) / chunk_num)
                summary_shards = []
                for i, c in enumerate(chunks(ctc, chunk_size)):
                    if debug_mode:
                        st.markdown(c)
                        size = len(get_tokenizer().tokenize(c))
                        st.write(i, "==> len(#) =", len(c), ', len(#tokens) =', size)
                        st.markdown("\n\n---\n")

                    my_bar.progress((i+1)/(chunk_num+1), text=progress_text + f" (第 {i+1} / {chunk_num} 部分)")

                    msg, tokens = paragraph_summary(c, t)
                    summary_shards.append(msg)
                    total_tokens += tokens

                    if i >= 5:
                        break
                
                my_bar.progress(0.9, text="正在聚合内容... 请稍后")
                # aggregate summary
                agg_summary, tokens = aggregate_summary(summary_shards)
                total_tokens += tokens
                summary = agg_summary

                if debug_mode:
                    st.sidebar.markdown("\n".join(summary_shards) + '\n\n --- \n\n')

            else:
                if debug_mode:
                    st.markdown(ctc)
                msg, tokens = paragraph_summary(ctc, t)
                summary += msg + '\n'
                total_tokens += tokens
        toc = time.time()
        estimate_rate = total_tokens / (toc - tic) * 60 if toc > tic else 0
        my_bar.progress(1.0, text=f'解析完成 （耗时`{int(toc-tic)}秒` 共计约`{int(total_tokens/1.845)}字符`）')
        st.markdown(summary)
        st.button('生成文章摘要卡片', disabled=True)
        st.sidebar.markdown(f"Rate: `{int(estimate_rate)} tokens/min`, Elapsed: `{int(toc - tic)}` seconds")
        st.sidebar.subheader('Summary:')
        st.sidebar.markdown(summary)

        # title_emoji = generate_title_emoji(title)
        # st.markdown(title_emoji)
        article_category = generate_article_category(summary)

        cn_words = len(list(filter(lambda w: '\u4e00' <= w <= '\u9fff', content)))
        en_words = (len(content) - cn_words) / 8

        image = generate_article_image(title, summary, url, article_category, round(cn_words + en_words), author)
        st.image(image, caption='已生成文章卡片，长按或右键保存')

        if toc - tic > 5 and invite_code_counter is not None:
            invite_code_counter.use_code(st.session_state['session_code'])
        
        # TDOD: add feedback button
    else:
        st.markdown("提示：url 格式类似 `https://mp.weixin.qq.com/s/...` ")