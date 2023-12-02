import json, logging, os
import aiohttp, requests
from urllib.parse import urlencode
from gradio_client import Client
import traceback
import edge_tts

from utils.common import Common
from utils.logger import Configure_logger
from utils.config import Config

class MY_TTS:
    def __init__(self, config_path):
        self.common = Common()
        self.config = Config(config_path)

        # 请求超时
        self.timeout = 60

        # 日志文件路径
        file_path = "./log/log-" + self.common.get_bj_time(1) + ".txt"
        Configure_logger(file_path)

        try:
            self.audio_out_path = self.config.get("play_audio", "out_path")

            if not os.path.isabs(self.audio_out_path):
                if not self.audio_out_path.startswith('./'):
                    self.audio_out_path = './' + self.audio_out_path
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error("请检查播放音频的音频输出路径配置！！！这将影响程序使用！")


    # 请求vits的api
    async def vits_api(self, data):
        try:
            if data["type"] == "vits":
                # API地址 "http://127.0.0.1:23456/voice/vits"
                API_URL = data["api_ip_port"] + '/voice/vits'
                data_json = {
                    "text": data["content"],
                    "id": data["id"],
                    "format": data["format"],
                    "lang": "ja",
                    "length": data["length"],
                    "noise": data["noise"],
                    "noisew": data["noisew"],
                    "max": data["max"]
                }
                
                if data["lang"] == "中文" or data["lang"] == "汉语":
                    data_json["lang"] = "zh"
                elif data["lang"] == "英文" or data["lang"] == "英语":
                    data_json["lang"] = "en"
                elif data["lang"] == "韩文" or data["lang"] == "韩语":
                    data_json["lang"] = "ko"
                elif data["lang"] == "日文" or data["lang"] == "日语":
                    data_json["lang"] = "ja"
                elif data["lang"] == "自动":
                    data_json["lang"] = "auto"
                else:
                    data_json["lang"] = "auto"
            elif data["type"] == "bert_vits2":
                # API地址 "http://127.0.0.1:23456/voice/bert-vits2"
                API_URL = data["api_ip_port"] + '/voice/bert-vits2'
                data_json = {
                    "text": data["content"],
                    "id": data["id"],
                    "format": data["format"],
                    "lang": "ja",
                    "length": data["length"],
                    "noise": data["noise"],
                    "noisew": data["noisew"],
                    "max": data["max"],
                    "sdp_radio": data["sdp_radio"]
                }
                
                if data["lang"] == "中文" or data["lang"] == "汉语":
                    data_json["lang"] = "zh"
                elif data["lang"] == "英文" or data["lang"] == "英语":
                    data_json["lang"] = "en"
                elif data["lang"] == "韩文" or data["lang"] == "韩语":
                    data_json["lang"] = "ko"
                elif data["lang"] == "日文" or data["lang"] == "日语":
                    data_json["lang"] = "ja"
                elif data["lang"] == "自动":
                    data_json["lang"] = "auto"
                else:
                    data_json["lang"] = "auto"

            # logging.info(f"data_json={data_json}")
            # logging.info(f"data={data}")

            url = f"{API_URL}?{urlencode(data_json)}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    response = await response.read()
                    # print(response)
                    file_name = 'vits_' + self.common.get_bj_time(4) + '.wav'
                    voice_tmp_path = self.common.get_new_audio_path(self.audio_out_path, file_name)
                    with open(voice_tmp_path, 'wb') as f:
                        f.write(response)
                    
                    return voice_tmp_path
        except aiohttp.ClientError as e:
            logging.error(f'vits请求失败，请检查您的vits-simple-api是否启动/配置是否正确，报错内容: {e}')
        except Exception as e:
            logging.error(f'vits未知错误，请检查您的vits-simple-api是否启动/配置是否正确，报错内容: {e}')
        
        return None


    
    # 请求VITS fast接口获取合成后的音频路径
    def vits_fast_api(self, data):
        try:
            # API地址
            API_URL = data["api_ip_port"] + '/run/predict/'

            data_json = {
                "fn_index":0,
                "data":[
                    "こんにちわ。",
                    "ikaros",
                    "日本語",
                    1
                ],
                "session_hash":"mnqeianp9th"
            }

            data_json["data"] = [data["content"], data["character"], data["language"], data["speed"]]

            logging.debug(f'data_json={data_json}')

            response = requests.post(url=API_URL, json=data_json, timeout=self.timeout)
            response.raise_for_status()  # 检查响应的状态码

            result = response.content
            ret = json.loads(result)

            file_path = ret["data"][1]["name"]

            new_file_path = self.common.move_file(file_path, os.path.join(self.audio_out_path, 'vits_fast_' + self.common.get_bj_time(4)), 'vits_fast_' + self.common.get_bj_time(4))

            return new_file_path
        except Exception as e:
            logging.error(f'vits-fast错误，请检查您的vits-fast推理程序是否启动/配置是否正确，报错内容: {e}')
            return None
    

    # 请求Edge-TTS接口获取合成后的音频路径
    async def edge_tts_api(self, data):
        try:
            file_name = 'edge_tts_' + self.common.get_bj_time(4) + '.mp3'
            voice_tmp_path = self.common.get_new_audio_path(self.audio_out_path, file_name)
            # voice_tmp_path = './out/' + self.common.get_bj_time(4) + '.mp3'
            # 过滤" '字符
            data["content"] = data["content"].replace('"', '').replace("'", '')
            # 使用 Edge TTS 生成回复消息的语音文件
            communicate = edge_tts.Communicate(text=data["content"], voice=data["voice"], rate=data["rate"], volume=data["volume"])
            await communicate.save(voice_tmp_path)

            return voice_tmp_path
        except Exception as e:
            logging.error(e)
            return None
        

    # 请求bark-gui的api
    def bark_gui_api(self, data):
        try:
            client = Client(data["api_ip_port"])
            result = client.predict(
                data["content"],	# str  in 'Input Text' Textbox component
                data["spk"],	# str (Option from: ['None', 'announcer', 'custom\\MeMyselfAndI', 'de_speaker_0', 'de_speaker_1', 'de_speaker_2', 'de_speaker_3', 'de_speaker_4', 'de_speaker_5', 'de_speaker_6', 'de_speaker_7', 'de_speaker_8', 'de_speaker_9', 'en_speaker_0', 'en_speaker_1', 'en_speaker_2', 'en_speaker_3', 'en_speaker_4', 'en_speaker_5', 'en_speaker_6', 'en_speaker_7', 'en_speaker_8', 'en_speaker_9', 'es_speaker_0', 'es_speaker_1', 'es_speaker_2', 'es_speaker_3', 'es_speaker_4', 'es_speaker_5', 'es_speaker_6', 'es_speaker_7', 'es_speaker_8', 'es_speaker_9', 'fr_speaker_0', 'fr_speaker_1', 'fr_speaker_2', 'fr_speaker_3', 'fr_speaker_4', 'fr_speaker_5', 'fr_speaker_6', 'fr_speaker_7', 'fr_speaker_8', 'fr_speaker_9', 'hi_speaker_0', 'hi_speaker_1', 'hi_speaker_2', 'hi_speaker_3', 'hi_speaker_4', 'hi_speaker_5', 'hi_speaker_6', 'hi_speaker_7', 'hi_speaker_8', 'hi_speaker_9', 'it_speaker_0', 'it_speaker_1', 'it_speaker_2', 'it_speaker_3', 'it_speaker_4', 'it_speaker_5', 'it_speaker_6', 'it_speaker_7', 'it_speaker_8', 'it_speaker_9', 'ja_speaker_0', 'ja_speaker_1', 'ja_speaker_2', 'ja_speaker_3', 'ja_speaker_4', 'ja_speaker_5', 'ja_speaker_6', 'ja_speaker_7', 'ja_speaker_8', 'ja_speaker_9', 'ko_speaker_0', 'ko_speaker_1', 'ko_speaker_2', 'ko_speaker_3', 'ko_speaker_4', 'ko_speaker_5', 'ko_speaker_6', 'ko_speaker_7', 'ko_speaker_8', 'ko_speaker_9', 'pl_speaker_0', 'pl_speaker_1', 'pl_speaker_2', 'pl_speaker_3', 'pl_speaker_4', 'pl_speaker_5', 'pl_speaker_6', 'pl_speaker_7', 'pl_speaker_8', 'pl_speaker_9', 'pt_speaker_0', 'pt_speaker_1', 'pt_speaker_2', 'pt_speaker_3', 'pt_speaker_4', 'pt_speaker_5', 'pt_speaker_6', 'pt_speaker_7', 'pt_speaker_8', 'pt_speaker_9', 'ru_speaker_0', 'ru_speaker_1', 'ru_speaker_2', 'ru_speaker_3', 'ru_speaker_4', 'ru_speaker_5', 'ru_speaker_6', 'ru_speaker_7', 'ru_speaker_8', 'ru_speaker_9', 'speaker_0', 'speaker_1', 'speaker_2', 'speaker_3', 'speaker_4', 'speaker_5', 'speaker_6', 'speaker_7', 'speaker_8', 'speaker_9', 'tr_speaker_0', 'tr_speaker_1', 'tr_speaker_2', 'tr_speaker_3', 'tr_speaker_4', 'tr_speaker_5', 'tr_speaker_6', 'tr_speaker_7', 'tr_speaker_8', 'tr_speaker_9', 'v2\\de_speaker_0', 'v2\\de_speaker_1', 'v2\\de_speaker_2', 'v2\\de_speaker_3', 'v2\\de_speaker_4', 'v2\\de_speaker_5', 'v2\\de_speaker_6', 'v2\\de_speaker_7', 'v2\\de_speaker_8', 'v2\\de_speaker_9', 'v2\\en_speaker_0', 'v2\\en_speaker_1', 'v2\\en_speaker_2', 'v2\\en_speaker_3', 'v2\\en_speaker_4', 'v2\\en_speaker_5', 'v2\\en_speaker_6', 'v2\\en_speaker_7', 'v2\\en_speaker_8', 'v2\\en_speaker_9', 'v2\\es_speaker_0', 'v2\\es_speaker_1', 'v2\\es_speaker_2', 'v2\\es_speaker_3', 'v2\\es_speaker_4', 'v2\\es_speaker_5', 'v2\\es_speaker_6', 'v2\\es_speaker_7', 'v2\\es_speaker_8', 'v2\\es_speaker_9', 'v2\\fr_speaker_0', 'v2\\fr_speaker_1', 'v2\\fr_speaker_2', 'v2\\fr_speaker_3', 'v2\\fr_speaker_4', 'v2\\fr_speaker_5', 'v2\\fr_speaker_6', 'v2\\fr_speaker_7', 'v2\\fr_speaker_8', 'v2\\fr_speaker_9', 'v2\\hi_speaker_0', 'v2\\hi_speaker_1', 'v2\\hi_speaker_2', 'v2\\hi_speaker_3', 'v2\\hi_speaker_4', 'v2\\hi_speaker_5', 'v2\\hi_speaker_6', 'v2\\hi_speaker_7', 'v2\\hi_speaker_8', 'v2\\hi_speaker_9', 'v2\\it_speaker_0', 'v2\\it_speaker_1', 'v2\\it_speaker_2', 'v2\\it_speaker_3', 'v2\\it_speaker_4', 'v2\\it_speaker_5', 'v2\\it_speaker_6', 'v2\\it_speaker_7', 'v2\\it_speaker_8', 'v2\\it_speaker_9', 'v2\\ja_speaker_0', 'v2\\ja_speaker_1', 'v2\\ja_speaker_2', 'v2\\ja_speaker_3', 'v2\\ja_speaker_4', 'v2\\ja_speaker_5', 'v2\\ja_speaker_6', 'v2\\ja_speaker_7', 'v2\\ja_speaker_8', 'v2\\ja_speaker_9', 'v2\\ko_speaker_0', 'v2\\ko_speaker_1', 'v2\\ko_speaker_2', 'v2\\ko_speaker_3', 'v2\\ko_speaker_4', 'v2\\ko_speaker_5', 'v2\\ko_speaker_6', 'v2\\ko_speaker_7', 'v2\\ko_speaker_8', 'v2\\ko_speaker_9', 'v2\\pl_speaker_0', 'v2\\pl_speaker_1', 'v2\\pl_speaker_2', 'v2\\pl_speaker_3', 'v2\\pl_speaker_4', 'v2\\pl_speaker_5', 'v2\\pl_speaker_6', 'v2\\pl_speaker_7', 'v2\\pl_speaker_8', 'v2\\pl_speaker_9', 'v2\\pt_speaker_0', 'v2\\pt_speaker_1', 'v2\\pt_speaker_2', 'v2\\pt_speaker_3', 'v2\\pt_speaker_4', 'v2\\pt_speaker_5', 'v2\\pt_speaker_6', 'v2\\pt_speaker_7', 'v2\\pt_speaker_8', 'v2\\pt_speaker_9', 'v2\\ru_speaker_0', 'v2\\ru_speaker_1', 'v2\\ru_speaker_2', 'v2\\ru_speaker_3', 'v2\\ru_speaker_4', 'v2\\ru_speaker_5', 'v2\\ru_speaker_6', 'v2\\ru_speaker_7', 'v2\\ru_speaker_8', 'v2\\ru_speaker_9', 'v2\\tr_speaker_0', 'v2\\tr_speaker_1', 'v2\\tr_speaker_2', 'v2\\tr_speaker_3', 'v2\\tr_speaker_4', 'v2\\tr_speaker_5', 'v2\\tr_speaker_6', 'v2\\tr_speaker_7', 'v2\\tr_speaker_8', 'v2\\tr_speaker_9', 'v2\\zh_speaker_0', 'v2\\zh_speaker_1', 'v2\\zh_speaker_2', 'v2\\zh_speaker_3', 'v2\\zh_speaker_4', 'v2\\zh_speaker_5', 'v2\\zh_speaker_6', 'v2\\zh_speaker_7', 'v2\\zh_speaker_8', 'v2\\zh_speaker_9', 'zh_speaker_0', 'zh_speaker_1', 'zh_speaker_2', 'zh_speaker_3', 'zh_speaker_4', 'zh_speaker_5', 'zh_speaker_6', 'zh_speaker_7', 'zh_speaker_8', 'zh_speaker_9']) in 'Voice' Dropdown component
                data["generation_temperature"],	# int | float (numeric value between 0.1 and 1.0) in 'Generation Temperature' Slider component
                data["waveform_temperature"],	# int | float (numeric value between 0.1 and 1.0) in 'Waveform temperature' Slider component
                data["end_of_sentence_probability"],	# int | float (numeric value between 0.0 and 0.5) in 'End of sentence probability' Slider component
                data["quick_generation"],	# bool  in 'Quick Generation' Checkbox component
                [],	# List[str]  in 'Detailed Generation Settings' Checkboxgroup component
                data["seed"],	# int | float  in 'Seed (default -1 = Random)' Number component
                data["batch_count"],	# int | float  in 'Batch count' Number component
                fn_index=3
            )

            new_file_path = self.common.move_file(result, os.path.join(self.audio_out_path, 'bark_gui_' + self.common.get_bj_time(4)), 'bark_gui_' + self.common.get_bj_time(4))

            return new_file_path
        except Exception as e:
            logging.error(f'bark_gui请求失败，请检查您的bark_gui是否启动/配置是否正确，报错内容: {e}')
            return None
    

    # 请求VALL-E-X的api
    def vall_e_x_api(self, data):
        try:
            client = Client(data["api_ip_port"])
            result = client.predict(
				data["content"],	# str in 'Text' Textbox component
				data["language"],	# str (Option from: ['auto-detect', 'English', '中文', '日本語', 'Mix']) in 'language' Dropdown component
				data["accent"],	# str (Option from: ['no-accent', 'English', '中文', '日本語']) in 'accent' Dropdown component
				data["voice_preset"],	# str (Option from: ['astraea', 'cafe', 'dingzhen', 'esta', 'ikaros', 'MakiseKurisu', 'mikako', 'nymph', 'rosalia', 'seel', 'sohara', 'sukata', 'tomoki', 'tomoko', 'yaesakura', '早见沙织', '神里绫华-日语']) in 'Voice preset' Dropdown component
				data["voice_preset_file_path"],	# str (filepath or URL to file) in 'parameter_46' File component
				fn_index=5
            )

            new_file_path = self.common.move_file(result[1], os.path.join(self.audio_out_path, 'vall_e_x_' + self.common.get_bj_time(4)), 'vall_e_x_' + self.common.get_bj_time(4))

            return new_file_path
        except Exception as e:
            logging.error(f'vall_e_x_api请求失败，请检查您的bark_gui是否启动/配置是否正确，报错内容: {e}')
            return None


    # 请求genshinvoice.top的api
    async def genshinvoice_top_api(self, text):
        url = 'https://genshinvoice.top/api'

        genshinvoice_top = self.config.get("genshinvoice_top")

        params = {
            'speaker': genshinvoice_top['speaker'],
            'text': text,
            'format': genshinvoice_top['format'],
            'length': genshinvoice_top['length'],
            'noise': genshinvoice_top['noise'],
            'noisew': genshinvoice_top['noisew'],
            'language': genshinvoice_top['language']
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=self.timeout) as response:
                    response = await response.read()
                    # voice_tmp_path = os.path.join(self.audio_out_path, 'genshinvoice_top_' + self.common.get_bj_time(4) + '.wav')
                    file_name = 'genshinvoice_top_' + self.common.get_bj_time(4) + '.wav'

                    voice_tmp_path = self.common.get_new_audio_path(self.audio_out_path, file_name)

                    with open(voice_tmp_path, 'wb') as f:
                        f.write(response)
                    
                    return voice_tmp_path
        except aiohttp.ClientError as e:
            logging.error(f'genshinvoice.top请求失败: {e}')
        except Exception as e:
            logging.error(f'genshinvoice.top未知错误: {e}')
        
        return None

    # 请求https://tts.ai-hobbyist.org/的api
    async def tts_ai_lab_top_api(self, text):
        url = 'https://tts.ai-lab.top'

        tts_ai_lab_top = self.config.get("tts_ai_lab_top")

        params = {
            "token": tts_ai_lab_top['token'],
            'speaker': tts_ai_lab_top['speaker'],
            'text': text,
            'sdp_ratio': tts_ai_lab_top['sdp_ratio'],
            'length': tts_ai_lab_top['length'],
            'noise': tts_ai_lab_top['noise'],
            'noisew': tts_ai_lab_top['noisew']
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=params, timeout=self.timeout) as response:
                    ret = await response.json()
                    logging.debug(ret)

                    file_url = ret["audio"]

                    async with session.get(file_url, timeout=self.timeout) as response:
                        if response.status == 200:
                            content = await response.read()

                            # voice_tmp_path = os.path.join(self.audio_out_path, 'tts_ai_lab_top_' + self.common.get_bj_time(4) + '.wav')
                            file_name = 'tts_ai_lab_top_' + self.common.get_bj_time(4) + '.wav'

                            voice_tmp_path = self.common.get_new_audio_path(self.audio_out_path, file_name)
                            
                            with open(voice_tmp_path, 'wb') as file:
                                file.write(content)

                            return voice_tmp_path
                        else:
                            logging.error(f'tts.ai-lab.top下载音频失败: {response.status}')
                            return None
        except aiohttp.ClientError as e:
            logging.error(f'tts.ai-lab.top请求失败: {e}')
        except Exception as e:
            logging.error(f'tts.ai-lab.top未知错误: {e}')
        
        return None

    # 请求OpenAI_TTS的api
    def openai_tts_api(self, data):
        try:
            if data["type"] == "huggingface":
                client = Client(data["api_ip_port"])
                result = client.predict(
                    data["content"],	# str in 'Text' Textbox component
                    data["model"],	# Literal[tts-1, tts-1-hd]  in 'Model' Dropdown component
                    data["voice"],	# Literal[alloy, echo, fable, onyx, nova, shimmer]  in 'Voice Options' Dropdown component
                    data["api_key"],	# str  in 'OpenAI API Key' Textbox component
                    api_name="/tts_enter_key"
                )

                new_file_path = self.common.move_file(result, os.path.join(self.audio_out_path, 'openai_tts_' + self.common.get_bj_time(4)), 'openai_tts_' + self.common.get_bj_time(4), "mp3")

                return new_file_path
            elif data["type"] == "api":
                from openai import OpenAI
                
                client = OpenAI(api_key=data["api_key"])

                response = client.audio.speech.create(
                    model=data["model"],
                    voice=data["voice"],
                    input=data["content"]
                )

                file_name = 'openai_tts_' + self.common.get_bj_time(4) + '.mp3'
                voice_tmp_path = self.common.get_new_audio_path(self.audio_out_path, file_name)

                response.stream_to_file(voice_tmp_path)

                return voice_tmp_path
        except Exception as e:
            logging.error(f'OpenAI_TTS请求失败: {e}')
            return None

    # 请求睿声AI的api
    async def reecho_ai_api(self, text):
        url = 'https://v1.reecho.ai/api/tts/simple-generate'

        reecho_ai = self.config.get("reecho_ai")
        
        headers = {  
            "Authorization": f"Bearer {reecho_ai['Authorization']}",  
            "Content-Type": "application/json"
        }

        params = {
            "model": reecho_ai['model'],
            'randomness': reecho_ai['randomness'],
            'stability_boost': reecho_ai['stability_boost'],
            'voiceId': reecho_ai['voiceId'],
            'text': text
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=params, timeout=self.timeout) as response:
                    ret = await response.json()
                    logging.debug(ret)

                    file_url = ret["data"]["audio"]

                    async with session.get(file_url, timeout=self.timeout) as response:
                        if response.status == 200:
                            content = await response.read()

                            # voice_tmp_path = os.path.join(self.audio_out_path, 'reecho_ai_' + self.common.get_bj_time(4) + '.wav')
                            file_name = 'reecho_ai_' + self.common.get_bj_time(4) + '.mp3'

                            voice_tmp_path = self.common.get_new_audio_path(self.audio_out_path, file_name)
                            
                            with open(voice_tmp_path, 'wb') as file:
                                file.write(content)

                            return voice_tmp_path
                        else:
                            logging.error(f'reecho.ai下载音频失败: {response.status}')
                            return None
        except aiohttp.ClientError as e:
            logging.error(f'reecho.ai请求失败: {e}')
        except Exception as e:
            logging.error(f'reecho.ai未知错误: {e}')
        
        return None
