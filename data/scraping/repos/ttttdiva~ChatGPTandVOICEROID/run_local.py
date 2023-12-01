#!usr/bin/env python
# -*- coding: utf-8 -*-

import os
from time import sleep

import openai
from dotenv import find_dotenv, load_dotenv

from character_manager import CharacterManager
from llm_manager import LLMManager
from tts_manager import TTSManager
from voice_recognizer import VoiceRecognizer

# 環境変数を読み込み
_ = load_dotenv(find_dotenv())
try:
    # OpenAIとGoogleのAPIキー
    openai.api_key = os.environ['OPENAI_API_KEY']
    google_api_key = os.environ['google_api_key']
    cx = os.environ['cx']
except KeyError as e:
    print(f"環境変数{e}が設定されていません。")
    input()

character_manager = CharacterManager()
llm_manager = LLMManager(character_manager.ai_chara, character_manager.ai_dialogues, google_api_key, cx, character_manager.ai_name)
voice_recognizer = VoiceRecognizer()
tts_manager = TTSManager("local", character_manager.tts_type, character_manager.voice_cid, character_manager.emo_coef, character_manager.emo_params)

def main():
    global llm_manager
    global tts_manager
    global voice_cid
    tts_manager.talk_message("起動しました！")
    # try:
    while True:
        # 音声認識
        voice_msg = voice_recognizer.voiceToText()

        if character_manager.char_select == False:
            # キャラクター選択処理
            if tts_manager.end_talk(voice_msg):
                tts_manager.talk_message("さようなら！")
                exit()

            # キャラを指定
            if any(name in voice_msg for name in character_manager.all_char_names):
                # キャラの情報を取得
                ai_name, ai_chara, ai_dialogues, voice_cid, greet, tts_type, emo_coef, emo_params = character_manager.get_character(voice_msg)
                tts_manager = TTSManager("local", tts_type, voice_cid, emo_coef, emo_params)
                
                # キャラプロンプトを読み込み
                llm_manager = LLMManager(ai_chara, ai_dialogues, google_api_key, cx, ai_name)
                
                tts_manager.talk_message(greet)
                character_manager.char_select = True
            else:
                print("名前以外が呼ばれた")
                continue
        else:
            # 会話メイン処理
            if tts_manager.end_talk(voice_msg):
                tts_manager.talk_message("End")
                print('talk終了')
                # 会話ログと要約を保存
                end = llm_manager.end_conversation()
                if end:
                    tts_manager.talk_message(end) # なんで書いたか忘れた処理
                
                if voice_msg in ["PCをシャットダウン", "おやすみ"]:
                    tts_manager.talk_message("おやすみなさい！")
                    os.system('shutdown /s /f /t 0')
                character_manager.char_select = False
                continue
            elif voice_msg == "前回の続き":
                tts_manager.talk_message("ちょっと待ってね！")
                
                # ログファイルから前回の会話を読み込んでmessagesに追加
                llm_manager.load_previous_chat()
                voice_msg = "今までどんなことを話していたっけ？30文字程度で教えて。"
            elif "検索して" in voice_msg:
                pass
            elif "チャットモード" in voice_msg:
                tts_manager.talk_message("Chatモードに切り替わりました。")
                llm_manager.switch_to_chat_mode()
                continue
            elif "アシスタントモード" in voice_msg:
                tts_manager.talk_message("Assistantsモードに切り替わりました。")
                llm_manager.switch_to_assistants_mode()
                continue
            elif tts_manager.hallucination(voice_msg):
                continue

            # GPTに対して返答を求める
            response_data = llm_manager.get_response(voice_msg)
            if isinstance(response_data, tuple):
                return_msg, emo_params = response_data
            else:
                return_msg = response_data
                emo_params = {}
            tts_manager.talk_message(return_msg, emo_params)
    # except Exception as e:
    #     tts_manager.talk_message("エラーが発生しました。")
    #     print('talk終了')
    #     # # 会話を保存
    #     # llm_manager.save_conversation(ai_name[0])
    #     print(e)


if __name__ == '__main__':
    main()
