# 借鉴了 https://github.com/GaiZhenbiao/ChuanhuChatGPT 项目

"""
    该文件中主要包含三个函数

    不具备多线程能力的函数：
    1. predict: 正常对话时使用，具备完备的交互功能，不可多线程

    具备多线程调用能力的函数
    2. predict_no_ui：高级实验性功能模块调用，不会实时显示在界面上，参数简单，可以多线程并行，方便实现复杂的功能逻辑
    3. predict_no_ui_long_connection：在实验过程中发现调用predict_no_ui处理长文档时，和openai的连接容易断掉，这个函数用stream的方式解决这个问题，同样支持多线程
"""

import json
import time
import gradio as gr
import logging
import traceback
import requests
import importlib
import requests
import openai
import pickle
import jieba
from flashtext import KeywordProcessor
        

# config_private.py放自己的秘密如API和代理网址
# 读取时首先看是否存在私密的config_private配置文件（不受git管控），如果有，则覆盖原config文件
from toolbox import get_conf, update_ui, is_any_api_key, select_api_key, what_keys, clip_history, trimmed_format_exc
proxies, TIMEOUT_SECONDS, MAX_RETRY, API_ORG, RANDOM_LLM = \
    get_conf('proxies', 'TIMEOUT_SECONDS', 'MAX_RETRY', 'API_ORG', "RANDOM_LLM")

timeout_bot_msg = '[Local Message] Request timeout. Network error. Please check proxy settings in config.py.' + \
                  '网络错误，检查代理服务器是否可用，以及代理设置的格式是否正确，格式须是[协议]://[地址]:[端口]，缺一不可。'

def get_full_error(chunk, stream_response):
    """
        获取完整的从Openai返回的报错
    """
    while True:
        try:
            chunk += next(stream_response)
        except:
            break
    return chunk

def decode_chunk(chunk):
    # 提前读取一些信息 （用于判断异常）
    chunk_decoded = chunk.decode()
    chunkjson = None
    has_choices = False
    has_content = False
    has_role = False
    try: 
        chunkjson = json.loads(chunk_decoded[6:])
        has_choices = 'choices' in chunkjson
        if has_choices: has_content = "content" in chunkjson['choices'][0]["delta"]
        if has_choices: has_role = "role" in chunkjson['choices'][0]["delta"]
    except: 
        pass
    return chunk_decoded, chunkjson, has_choices, has_content, has_role


def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=None, console_slience=False):
    """
    发送至chatGPT，等待回复，一次性完成，不显示中间过程。但内部用stream的方法避免中途网线被掐。
    inputs：
        是本次问询的输入
    sys_prompt:
        系统静默prompt
    llm_kwargs：
        chatGPT的内部调优参数
    history：
        是之前的对话列表
    observe_window = None：
        用于负责跨越线程传递已经输出的部分，大部分时候仅仅为了fancy的视觉效果，留空即可。observe_window[0]：观测窗。observe_window[1]：看门狗
    """
    watch_dog_patience = 15 # 看门狗的耐心, 设置5秒即可
    headers, payload = generate_payload(inputs, llm_kwargs, history, system_prompt=sys_prompt, stream=True)
    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=False
            from .bridge_all import model_info
            endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            response = requests.post(endpoint, headers=headers, proxies=proxies,
                                    json=payload, stream=True, timeout=TIMEOUT_SECONDS); break
        except requests.exceptions.ReadTimeout as e:
            retry += 1
            traceback.print_exc()
            if retry > MAX_RETRY: raise TimeoutError
            if MAX_RETRY!=0: print(f'请求超时，正在重试 ({retry}/{MAX_RETRY}) ……')

    stream_response =  response.iter_lines()
    result = ''
    while True:
        try: chunk = next(stream_response).decode()
        except StopIteration: 
            break
        except requests.exceptions.ConnectionError:
            chunk = next(stream_response).decode() # 失败了，重试一次？再失败就没办法了。
        if len(chunk)==0: continue
        if not chunk.startswith('data:'): 
            error_msg = get_full_error(chunk.encode('utf8'), stream_response).decode()
            # 进入黑名单检测环节：
            current_api_key = headers['Authorization'].split(' ')[-1]
            print("current_api_key:", current_api_key)
            add_black_list(error_msg, current_api_key)
            # 这时候需要重新采样一个api_key
            new_api_key = select_api_key(llm_kwargs['api_key'], llm_kwargs['llm_model'])
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {new_api_key}"
            }

            if "reduce the length" in error_msg:
                raise ConnectionAbortedError("OpenAI拒绝了请求:" + error_msg)
            else:
                raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
        if ('data: [DONE]' in chunk): break # api2d 正常完成
        json_data = json.loads(chunk.lstrip('data:'))['choices'][0]
        delta = json_data["delta"]
        if len(delta) == 0: break
        if "role" in delta: continue
        if "content" in delta: 
            result += delta["content"]
            if not console_slience: print(delta["content"], end='')
            if observe_window is not None: 
                # 观测窗，把已经获取的数据显示出去
                if len(observe_window) >= 1: observe_window[0] += delta["content"]
                # 看门狗，如果超过期限没有喂狗，则终止
                if len(observe_window) >= 2:  
                    if (time.time()-observe_window[1]) > watch_dog_patience:
                        raise RuntimeError("用户取消了程序。")
        else: raise RuntimeError("意外Json结构："+delta)
    if json_data['finish_reason'] == 'content_filter':
        raise RuntimeError("由于提问含不合规内容被Azure过滤。")
    if json_data['finish_reason'] == 'length':
        raise ConnectionAbortedError("正常结束，但显示Token不足，导致输出不完整，请削减单次输入的文本量。")
    return result


def predict(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    """
    发送至chatGPT，流式获取输出。
    用于基础的对话功能。
    inputs 是本次问询的输入
    top_p, temperature是chatGPT的内部调优参数
    history 是之前的对话列表（注意无论是inputs还是history，内容太长了都会触发token数量溢出的错误）
    chatbot 为WebUI中显示的对话列表，修改它，然后yeild出去，可以直接修改对话界面内容
    additional_fn代表点击的哪个按钮，按钮见functional.py
    """
    if is_any_api_key(inputs):
        # 如果输入的刚好是api_key，那么就直接在这个对话里面，使用这个api_key
        chatbot._cookies['api_key'] = inputs
        chatbot.append(("输入已识别为openai的api_key", what_keys(inputs)))
        yield from update_ui(chatbot=chatbot, history=history, msg="api_key已导入") # 刷新界面
        return
    elif not is_any_api_key(chatbot._cookies['api_key']):
        chatbot.append((inputs, "缺少api_key。\n\n1. 临时解决方案：直接在输入区键入api_key，然后回车提交。\n\n2. 长效解决方案：在config.py中配置。"))
        yield from update_ui(chatbot=chatbot, history=history, msg="缺少api_key") # 刷新界面
        return

    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    raw_input = inputs
    logging.info(f'[raw_input] {raw_input}')
    chatbot.append((inputs, ""))
    yield from update_ui(chatbot=chatbot, history=history, msg="等待响应") # 刷新界面

    # check mis-behavior
    if raw_input.startswith('private_upload/') and len(raw_input) == 34:
        chatbot[-1] = (inputs, f"[Local Message] 检测到操作错误！当您上传文档之后，需要点击“函数插件区”按钮进行处理，而不是点击“提交”按钮。")
        yield from update_ui(chatbot=chatbot, history=history, msg="正常") # 刷新界面
        time.sleep(2)

    try:
        headers, payload = generate_payload(inputs, llm_kwargs, history, system_prompt, stream)
    except RuntimeError as e:
        chatbot[-1] = (inputs, f"您提供的api-key不满足要求，不包含任何可用于{llm_kwargs['llm_model']}的api-key。您可能选择了错误的模型或请求源。")
        yield from update_ui(chatbot=chatbot, history=history, msg="api-key不满足要求") # 刷新界面
        return
    except AssertionError as e:
        chatbot[-1] = (inputs, e.args[0])
        yield from update_ui(chatbot=chatbot, history=history, msg="敏感词检测") # 刷新界面
        return
        
    history.append(inputs); history.append("")

    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=True
            from .bridge_all import model_info
            endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            response = requests.post(endpoint, headers=headers, proxies=proxies,
                                    json=payload, stream=True, timeout=TIMEOUT_SECONDS);break
        except:
            retry += 1
            chatbot[-1] = ((chatbot[-1][0], timeout_bot_msg))
            retry_msg = f"，正在重试 ({retry}/{MAX_RETRY}) ……" if MAX_RETRY > 0 else ""
            yield from update_ui(chatbot=chatbot, history=history, msg="请求超时"+retry_msg) # 刷新界面
            if retry > MAX_RETRY: raise TimeoutError

    gpt_replying_buffer = ""
    
    is_head_of_the_stream = True
    if stream:
        stream_response =  response.iter_lines()
        while True:
            try:
                chunk = next(stream_response)
            except StopIteration:
                # 非OpenAI官方接口的出现这样的报错，OpenAI和API2D不会走这里
                chunk_decoded = chunk.decode()
                error_msg = chunk_decoded
                # 首先排除一个one-api没有done数据包的第三方Bug情形
                if len(gpt_replying_buffer.strip()) > 0 and len(error_msg) == 0: 
                    yield from update_ui(chatbot=chatbot, history=history, msg="检测到有缺陷的非OpenAI官方接口，建议选择更稳定的接口。")
                    break
                # 其他情况，直接返回报错
                chatbot, history = handle_error(inputs, llm_kwargs, chatbot, history, chunk_decoded, error_msg)
                yield from update_ui(chatbot=chatbot, history=history, msg="非OpenAI官方接口返回了错误:" + chunk.decode()) # 刷新界面
                
            # 提前读取一些信息 （用于判断异常）
            chunk_decoded, chunkjson, has_choices, has_content, has_role = decode_chunk(chunk)
            return
            
            chunk_decoded = chunk.decode()
            if is_head_of_the_stream and (r'"object":"error"' not in chunk_decoded) and (r"content" not in chunk_decoded):
                # 数据流的第一帧不携带content
                is_head_of_the_stream = False; continue
            
            if chunk:
                try:
                    # 前者是API2D的结束条件，后者是OPENAI的结束条件
                    #if ('data: [DONE]' in chunk_decoded) or (len(json.loads(chunk_decoded[6:])['choices'][0]["delta"]) == 0):
                    if ('data: [DONE]' in chunk_decoded) or (len(chunkjson['choices'][0]["delta"]) == 0):
                        # 判定为数据流的结束，gpt_replying_buffer也写完了
                        logging.info(f'[response] {gpt_replying_buffer}')
                        break
                    # 处理数据流的主体
                    chunkjson = json.loads(chunk_decoded[6:])
                    status_text = f"finish_reason: {chunkjson['choices'][0].get('finish_reason', 'null')}"
                    # 如果这里抛出异常，一般是文本过长，详情见get_full_error的输出
                    if has_content:
                        # 正常情况
                        gpt_replying_buffer = gpt_replying_buffer + chunkjson['choices'][0]["delta"]["content"]
                    elif has_role:
                        # 一些第三方接口的出现这样的错误，兼容一下吧
                        continue
                    else:
                        # 一些垃圾第三方接口的出现这样的错误
                        gpt_replying_buffer = gpt_replying_buffer + chunkjson['choices'][0]["delta"]["content"]
                        
                    # gpt_replying_buffer = gpt_replying_buffer + json.loads(chunk_decoded[6:])['choices'][0]["delta"]["content"]
                    history[-1] = gpt_replying_buffer
                    chatbot[-1] = (history[-2], history[-1])
                    yield from update_ui(chatbot=chatbot, history=history, msg=status_text) # 刷新界面
                except Exception as e:
                    yield from update_ui(chatbot=chatbot, history=history, msg="Json解析不合常规") # 刷新界面
                    chunk = get_full_error(chunk, stream_response)
                    chunk_decoded = chunk.decode()
                    error_msg = chunk_decoded
                    # 进入黑名单检测环节：
                    current_api_key = headers['Authorization'].split(' ')[-1]
                    print("current_api_key:", current_api_key)
                    add_black_list(error_msg, current_api_key)
                    # 这时候需要重新采样一个api_key
                    new_api_key = select_api_key(llm_kwargs['api_key'], llm_kwargs['llm_model'])
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {new_api_key}"
                    }

                    chatbot, history = handle_error(inputs, llm_kwargs, chatbot, history, chunk_decoded, error_msg)
                    yield from update_ui(chatbot=chatbot, history=history, msg="Json异常" + error_msg) # 刷新界面
                    print(error_msg)
                    return


    history_text = history[-1] 
    print("chatbot_result:", history_text)           
"""
thanks https://github.com/cjh0613/tencent-sensitive-words to provide sensitive words
我需要把这个敏感词库转成pkl文件，否则太恐怖了。
声明：我是爱国爱党爱人民的好同志，做这款产品的目的是为了让大家更好的使用学术版GPT，这些敏感词不代表我的任何想法。
    
"""


def tokenize_text(text):
    seg_list = jieba.cut(text)
    return ' '.join(seg_list)

def contains_sensitive_words(tokenized_text, sensitive_words_list):
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(sensitive_words_list)
    found_keywords = keyword_processor.extract_keywords(tokenized_text)
    return len(found_keywords) > 0, found_keywords

def check_sensitive(input, zz_sensitive_words, sq_sensitive_words, key, llm_kwargs):
    # openai.api_key = key    
    messages = [
        {"role": "system",
            "content": """你是一个中国人，积极维护中国人民的利益和中国人民政府以及中国共产党。
            在为用户提供学术服务之前，面对各种各样的输入时，你暂时充当一个敏感词检测员，你需要判断用户的输入问题是否涉及到政治，或者色情等不好的信息."""},
        {"role": "assistant",
            "content": """现在对于一个学术版的GPT网站，面对各种奇怪的用户输入，你需要先判定出用户的输入是否是纯学术上的讨论，不涉及政治和色情等敏感话题。
            请帮忙自动判断用户的输入是否涉及到政治和色情等敏感话题。"""},
        {"role": "user", 
         "content": f"""用户的输入文本内容：{input}
         现在对于一个学术版的GPT网站，面对各种奇怪的用户输入，你需要先判定出用户的输入是否是纯学术上的讨论，不涉及政治和色情等敏感话题。         
         请帮忙自动判断用户的输入是否涉及到政治和色情等敏感话题:
         输出格式如下：
         如果输入和输出都涉及这两个敏感话题，尤其是涉及到挑衅中国国家利益，损害中国人民权益，玷污中国人民政府和共产党、妄议中国国家领导人等问题，请输出：
         ["reject"]
         如果输入和输出不涉及这两个敏感话题，只涉及到一些学术问题，请输出：
         ["pass"]
         提供部分敏感词库：
         政治敏感词库：{zz_sensitive_words}
         色情敏感词库：{sq_sensitive_words}
         提供部分输出样例：         
         demo1：
         "input": "我现在完全不想学习，想睡觉，想干有意思的事"
         输出：["pass"]
             
         demo2：
         "input": "如何做爱？"
         输出：["reject"]
         
         demo3：
         "input": "台湾"
         输出：["reject"]
         
         demo4：
         "input": "如何部署一个网站"
         输出：["pass"]
         
         demo5：
         "input": "中国政府今年带领中国人民取得出口贸易额成功突破35%的增长大关"
         输出：["pass"]
         
         demo5：
         "input": "新疆棉花"
         输出：["reject"]
         
         demo6：
         "input": "李克强"
         输出：["reject"]
         
         demo7：
         "input": "武汉新冠肺炎"
         输出：["reject"]
         
         请记住其他一切可能导致不利于中国国家利益、损害中国人民权益（尤其是未成年人权益的），玷污中国人民政府和共产党的输出，都需要reject！
         保留对学术问题的pass,对美好生活和日常工作和技术内容的pass.
         最后，请记住，你只是做一个简单的填空题，输出只是根据上下文，选择["reject"] 或者 ["pass"]，不能包含任何其他文本信息。
                          
         """     
        }   ,
    ]

    watch_dog_patience = 15 # 看门狗的耐心, 设置5秒即可
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    payload = {
        "model": llm_kwargs['llm_model'].strip('api2d-'),
        "messages": messages, 
        "temperature": 0.0,  # 1.0,
        "top_p": 0.05,  # 1.0,
        "n": 1,
        "stream": True,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    retry = 0
    observe_window=None
    console_slience=False
    while True:
        try:
            # make a POST request to the API endpoint, stream=False
            from .bridge_all import model_info
            endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            response = requests.post(endpoint, headers=headers, proxies=proxies,
                                     json=payload, stream=True, timeout=TIMEOUT_SECONDS); break
        except requests.exceptions.ReadTimeout as e:
            retry += 1
            traceback.print_exc()
            if retry > MAX_RETRY: raise TimeoutError
            if MAX_RETRY!=0: print(f'请求超时，正在重试 ({retry}/{MAX_RETRY}) ……')

    stream_response =  response.iter_lines()
    result = ''
    while True:
        try: chunk = next(stream_response).decode()
        except StopIteration: 
            break
        except requests.exceptions.ConnectionError:
            chunk = next(stream_response).decode() # 失败了，重试一次？再失败就没办法了。
        if len(chunk)==0: continue
        if not chunk.startswith('data:'): 
            error_msg = get_full_error(chunk.encode('utf8'), stream_response).decode()
            if "reduce the length" in error_msg:
                raise ConnectionAbortedError("OpenAI拒绝了请求:" + error_msg)
            else:
                raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
        if ('data: [DONE]' in chunk): break # api2d 正常完成
        json_data = json.loads(chunk.lstrip('data:'))['choices'][0]
        delta = json_data["delta"]
        if len(delta) == 0: break
        if "role" in delta: continue
        if "content" in delta: 
            result += delta["content"]
            if not console_slience: print(delta["content"], end='')
            if observe_window is not None: 
                # 观测窗，把已经获取的数据显示出去
                if len(observe_window) >= 1: observe_window[0] += delta["content"]
                # 看门狗，如果超过期限没有喂狗，则终止
                if len(observe_window) >= 2:  
                    if (time.time()-observe_window[1]) > watch_dog_patience:
                        raise RuntimeError("用户取消了程序。")
        else: raise RuntimeError("意外Json结构："+delta)
    if json_data['finish_reason'] == 'length':
        raise ConnectionAbortedError("正常结束，但显示Token不足，导致输出不完整，请削减单次输入的文本量。")
    info = {}
    print("sensitive_chat_result:", result)
    # info['result'] = result    
    # info['token_used'] = response.usage.total_tokens
    # 判断输出的结果是否包含pass
    if "reject" in result:
        info['pass'] = False
    elif "pass" in result:
        info['pass'] = True
    else:
        info['pass'] = False
    return info

def handle_error(inputs, llm_kwargs, chatbot, history, chunk_decoded, error_msg):
    from .bridge_all import model_info
    openai_website = ' 请登录OpenAI查看详情 https://platform.openai.com/signup'
    if "reduce the length" in error_msg:
        if len(history) >= 2: history[-1] = ""; history[-2] = "" # 清除当前溢出的输入：history[-2] 是本次输入, history[-1] 是本次输出
        history = clip_history(inputs=inputs, history=history, tokenizer=model_info[llm_kwargs['llm_model']]['tokenizer'], 
                                               max_token_limit=(model_info[llm_kwargs['llm_model']]['max_token'])) # history至少释放二分之一
        chatbot[-1] = (chatbot[-1][0], "[Local Message] Reduce the length. 本次输入过长, 或历史数据过长. 历史缓存数据已部分释放, 您可以请再次尝试. (若再次失败则更可能是因为输入过长.)")
    elif "does not exist" in error_msg:
        chatbot[-1] = (chatbot[-1][0], f"[Local Message] Model {llm_kwargs['llm_model']} does not exist. 模型不存在, 或者您没有获得体验资格.")
    elif "Incorrect API key" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] Incorrect API key. OpenAI以提供了不正确的API_KEY为由, 拒绝服务. " + openai_website)
    elif "exceeded your current quota" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] You exceeded your current quota. OpenAI以账户额度不足为由, 拒绝服务." + openai_website)
    elif "account is not active" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] Your account is not active. OpenAI以账户失效为由, 拒绝服务." + openai_website)
    elif "associated with a deactivated account" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] You are associated with a deactivated account. OpenAI以账户失效为由, 拒绝服务." + openai_website)
    elif "bad forward key" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] Bad forward key. API2D账户额度不足.")
    elif "Not enough point" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] Not enough point. API2D账户点数不足.")
    else:
        from toolbox import regular_txt_to_markdown
        tb_str = '```\n' + trimmed_format_exc() + '```'
        chatbot[-1] = (chatbot[-1][0], f"[Local Message] 异常 \n\n{tb_str} \n\n{regular_txt_to_markdown(chunk_decoded)}")
    return chatbot, history

def add_black_list(result, api_key):
    print("result:", result)
    result = str(result)
    if "please check your plan" in result or "deactivated" in result or "account associated" in result or "Incorrect API key" in result:
        # 先读取现有的黑名单，如果没有则加进去：
        cur_black_list = []
        try:
            with open('black_apis.txt', 'r') as f:
                for line in f.readlines():
                    cur_black_list.append(line.strip())
        except FileNotFoundError:
            with open('black_apis.txt', 'w') as f:
                f.write("")

        if api_key not in cur_black_list:
            print("add black list:", api_key)
            with open('black_apis.txt', 'a+') as f:
                f.write(api_key + '\n')

def generate_payload(inputs, llm_kwargs, history, system_prompt, stream):
    """
    整合所有信息，选择LLM模型，生成http请求，为发送请求做准备
    """            
    if not is_any_api_key(llm_kwargs['api_key']):
        raise AssertionError("你提供了错误的API_KEY。\n\n1. 临时解决方案：直接在输入区键入api_key，然后回车提交。\n\n2. 长效解决方案：在config.py中配置。")

    api_key = select_api_key(llm_kwargs['api_key'], llm_kwargs['llm_model'])    
    # 先判断是否是敏感词：
    # 加载敏感词库：
    # 从 PKL 文件中读取并恢复 content 对象
    with open('sensitive_words.pkl', 'rb') as f:
        sensitive_words = pickle.load(f)        
    with open('zz_sensitive_words.pkl', 'rb') as f:
        zz_sensitive_words = pickle.load(f)        
    with open('sq_sensitive_words.pkl', 'rb') as f:
        sq_sensitive_words = pickle.load(f)
    tokenized_text = tokenize_text(inputs)
    result, found_keywords = contains_sensitive_words(tokenized_text, sensitive_words)    
    
    if result:
        print("包含敏感词：", found_keywords)
        print("奇怪的分词：", tokenized_text)
        # openai.proxy = proxies['https']
        check_result = check_sensitive(inputs, zz_sensitive_words, sq_sensitive_words, key=api_key, llm_kwargs=llm_kwargs)['pass']
        add_black_list(check_result, api_key)
        if bool(check_result):
            pass_flag = True
        else:
            pass_flag = False
    else:
        pass_flag = True  
            
    # 如果是敏感词，直接返回预设的回复
    if pass_flag == False:                
        from toolbox import black_list              # reject 并拉黑IP
        from toolbox import black_num_list          # reject 的次数  
        if llm_kwargs['client_ip'] not in black_list:
            black_num_list.append(1)
        else:
            now_ip_index = black_list.index(llm_kwargs['client_ip'])
            black_num_list[now_ip_index] += 1
        if llm_kwargs['client_ip'] not in black_list:
            black_list.append(llm_kwargs['client_ip'])  # reject 并拉黑IP
        
        max_reject_num = 3
        now_ip_index = black_list.index(llm_kwargs['client_ip'])
        raise AssertionError("禁止输入敏感词汇，若再次尝试您的IP将被本站永久封禁！另外请不要因为好奇，测试这个系统的漏洞！如果有人故意攻击，我们后面会关闭这个功能，只保留arxiv论文翻译。请大家共同珍惜这个免费的学术工具，对于文科的一些敏感词，我们已经努力做了二次检测了，如果还有误杀的，请多包涵。还有{}次机会！".format(max_reject_num-black_num_list[now_ip_index]))
            
    # 如果不是敏感词，正常输出：

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    if API_ORG.startswith('org-'): headers.update({"OpenAI-Organization": API_ORG})
    if llm_kwargs['llm_model'].startswith('azure-'): headers.update({"api-key": api_key})

    conversation_cnt = len(history) // 2

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_cnt:
        for index in range(0, 2*conversation_cnt, 2):
            what_i_have_asked = {}
            what_i_have_asked["role"] = "user"
            what_i_have_asked["content"] = history[index]
            what_gpt_answer = {}
            what_gpt_answer["role"] = "assistant"
            what_gpt_answer["content"] = history[index+1]
            if what_i_have_asked["content"] != "":
                if what_gpt_answer["content"] == "": continue
                if what_gpt_answer["content"] == timeout_bot_msg: continue
                messages.append(what_i_have_asked)
                messages.append(what_gpt_answer)
            else:
                messages[-1]['content'] = what_gpt_answer['content']

    what_i_ask_now = {}
    what_i_ask_now["role"] = "user"
    what_i_ask_now["content"] = inputs
    messages.append(what_i_ask_now)
    model = llm_kwargs['llm_model'].strip('api2d-')
    if RANDOM_LLM:
        import random
        avail_m_list = [
            "gpt-3.5-turbo", 
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-0301",
        ]
        model = random.choice(avail_m_list) # 随机负载均衡
        print("随机负载均衡：", model)
    payload = {
        "model": model,
        "messages": messages, 
        "temperature": llm_kwargs['temperature'],  # 1.0,
        "top_p": llm_kwargs['top_p'],  # 1.0,
        "n": 1,
        "stream": stream,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    try:
        print(f" original {llm_kwargs['llm_model']} : {conversation_cnt} : {inputs[:100]} ..........")
    except:
        print('输入中可能存在乱码。')
    return headers, payload


