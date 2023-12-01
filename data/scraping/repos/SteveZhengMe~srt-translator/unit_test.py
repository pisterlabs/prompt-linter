import unittest
import deepl
import openai
import srt

from libraries import DeepLUtil
from libraries import OpenAIUtil
from libraries import SRTTranslator
from app import init_conf, create_engine, scan_folder

class TestStringMethods(unittest.TestCase):
    # setup
    def setUp(self):
        self.conf = init_conf()
        # find the DeepLUtil in the list
        self.deepl_engine = list(filter(lambda x: isinstance(x, DeepLUtil), create_engine(self.conf)))
        # find the OpenAIUtil in the list
        self.openai_engine = list(filter(lambda x: isinstance(x, OpenAIUtil), create_engine(self.conf)))
    
    def test_load_env(self):
        self.assertNotEquals(len(self.conf), 0)
        self.assertNotEquals(len(self.deepl_engine), 0)
        self.assertNotEquals(len(self.openai_engine), 0)
        self.assertEquals(self.conf["deepl_key"][-2:],"fx")
        self.assertEquals(self.conf["openai_key"][:2],"sk")

    def test_deepL_translate(self):
        deepl_translator =  deepl.Translator(self.conf["deepl_key"])
        result = deepl_translator.translate_text(
            ["Hello, Tom. <br> In today’s [globalized] world, language barriers are a challenge that businesses and individuals often face.", "I speak Chinese"], 
            target_lang=self.conf["target_language"][0:2]
        )
        self.assertEquals(result[0].text,"你好，汤姆。<br> 在当今[全球化]的世界里，语言障碍是企业和个人经常面临的挑战。")
        self.assertEquals(result[1].text,"我会说中文")
    
    def test_deepl_get_usage(self):
        deepl_translator =  deepl.Translator(self.conf["deepl_key"])
        # get usage
        usage = deepl_translator.get_usage()
        print(usage.character.count)
        self.assertTrue(usage.character.count < usage.character.limit)
        
    def test_openai_translate(self):
        openai.api_key = self.conf["openai_key"]
        target_language = "zh_CN"
        text = "<p>Hello, <br>Tom. </p><p>In today’s [globalized] world, language barriers are a challenge that businesses and individuals often face.</p><p>I speak Chinese</p>"
        
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": f"{self.conf['openai_system_prompt']} {target_language}."},
                {"role": "user", "content": f"{self.conf['openai_user_prompt_default']} {text}"}
            ],
            temperature=0 
        )
        #print(chat_completion.choices[0].message.content.strip())
        self.assertEquals(chat_completion.choices[0].message.content.strip(), "<p>你好，<br>Tom。</p><p>在今天的[全球化]世界中，语言障碍是企业和个人经常面临的挑战。</p><p>我会说中文</p>")
        
    def test_parse_srt(self):
        with open("test-data/test.srt") as srt_file:
            subtitles = list(srt.parse(srt_file.read()))
            self.assertEquals(len(subtitles), 150)
            self.assertEqual(subtitles[0].content.replace("\n","||"),"Downloaded from||YTS.MX")
    
    ########################################################################
    def test_SRTTranslator(self):
        srt_parser = SRTTranslator("test-data/test.srt", self.conf)
        self.assertEquals(len(srt_parser.subtitles), 106)
        
    def test_DeepLUtil(self):
        deepl_util = self.openai_engine[0]
        self.assertTrue(deepl_util.is_available())
        self.assertEquals(deepl_util.translate(["Hello, Tom. || language barriers are a [challenge] that businesses and individuals often face.", "I speak Chinese"]),['你好，Tom。|| 语言障碍是企业和个人经常面临的[挑战]。', '我会说中文'])
        
    def test_OpenAIUtil(self):
        openai_util = self.openai_engine[0]
        self.assertTrue(openai_util.is_available())
        self.assertEquals(openai_util.translate(["Hello, Tom.|| In today’s [globalized] world, language barriers are a challenge that businesses and individuals often face.", "I speak Chinese."]),["你好，Tom。|| 在今天的[全球化]世界中，语言障碍是企业和个人经常面临的挑战。","我会说中文。"])
        
    def test_integrate(self):
        srt_parser = SRTTranslator("test-data/test.srt", self.conf)
        engine = []
        engine.extend(self.deepl_engine)
        engine.extend(self.openai_engine)
        
        srt_parser.translate(engine)
        srt_parser.save()
    
    def test_scan(self):
        scan_folder("./data/subs", "./data", "Chinese", "English", "y")
