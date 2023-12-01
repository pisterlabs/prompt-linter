import json
import os
import random
import re
import time
from pathlib import Path

import openai

from characterAI import CharacterAI, ChatController, OpenAILLM, contextDB_json
from utils.conversationRule import ConversationTimingChecker
from utils.FIFOPlayer import FIFOPlayer
from utils.getYoutubeChat import YoutubeChat
from utils.playWaveManager import WavQueuePlayer
from utils.tachieViewer import TachieViewer
from utils.textEdit import remove_chars
from utils.voicevoxUtils import makeWaveFile, text2stream, voicevoxHelthCheck
from utils.wavePlayerWithVolume import WavPlayerWithVolume


def main():
    if not voicevoxHelthCheck():
        return
    try:
        yt_url = 'https://www.youtube.com/watch?v=RwI1m_okebY' # youtubeのURL
        youtubeChat = YoutubeChat(yt_url)
        
        # キャラクターがwavファイルを作成する
        # キャラクターのwavファイルを順番に再生する
        # 立ち絵を動かす
        fifoPlayer = FIFOPlayer()
        fifoPlayer.playWithFIFO()

        # imagesDirPath = Path('characterConfig/test/images')
        # tachieViewer = TachieViewer(imagesDirPath)
        # tachieViewer.play()

        LLM = OpenAILLM()
        ruminesAI = CharacterAI(LLM, "ルミネス", contextDB_json,
                               13, 1.2, fifoPlayer, TachieViewer, 'rumines')
        rianAI = CharacterAI(LLM, "リアン", contextDB_json,
                              6, 1.2, fifoPlayer, TachieViewer, 'rian')
        ranAI = CharacterAI(LLM, "ラン", contextDB_json,
                               8, 1.2, fifoPlayer, TachieViewer, 'ran')
        neonAI = CharacterAI(
            LLM, "ネオン", contextDB_json, 7, 1.2, fifoPlayer, TachieViewer, 'neon')

        characterAIs = [ruminesAI, rianAI, ranAI, neonAI]

        conversationTimingChecker = ConversationTimingChecker()
        chatController = ChatController(characterAIs)
        chatController.initContextAll()
        chatController.addContextAll(
            'system', "[場面説明]\nあなたたちはyoutubeの配信者で,今デビュー配信中です。")

        print("start")
        time.sleep(15)
        chatController.postCharacterChat(rianAI, "さあ、はじまりました～")
        time.sleep(1)
        chatController.postCharacterChat(neonAI, "やるのだ～")
        time.sleep(1)
        chatController.postCharacterChat(ruminesAI, "いくぞっ！！！")
        time.sleep(1)
        chatController.postCharacterChat(ranAI, "皆さん、まじめにお願いしますよ！")
        time.sleep(1)
        
        chatController.postCharacterChat(ranAI, "はい、私たち、KyaraAIのデビュー配信です！")
        time.sleep(1)
        
        chatController.postCharacterChat(ranAI, "今日は、お忙しいなか配信を見てくださって、ありがとうございます！まずは一人ずつ自己紹介をさせていただきますね。一人目は、リアンさんです！")
        time.sleep(1)
        chatController.postCharacterChat(rianAI, "はいはい！KyaraAI所属のAIVtuber、金色と紫の髪色がチャームポイントのゴスロリメイド、リアンです！みなさんよろしくね！じゃあ、次はルミネス！")
        time.sleep(1)
        chatController.postCharacterChat(ruminesAI, "心に刻むがよい。オレの名はルミネス・アンドルファー、皇国エルドレアの侯爵だ。オレの事はルミネス様と呼ぶがよい。よろしくな！")
        time.sleep(1)
        chatController.postCharacterChat(neonAI, "次は僕なのだ～。KyaraAI所属のAIVtuber、ショートヘアの黒髪とピンクの光るネオンが特徴の未来からやってきたアンドロイド美少女、ネオンなのだ！みんな、よろしくなのだ！")
        time.sleep(1)
        chatController.postCharacterChat(ranAI, "そして最後は～、あたし、東京ギャルの花京院ランだよ★よろしくね！")
        time.sleep(1)
        
        chatController.postCharacterChat(ranAI, "じゃあ、まずは、みんなの趣味や好きなものを聞いてみようかな！")
        # chatController.addContextAll('system', "[場面説明]\n趣味や好きなものについて話す。")
        
        chatController.getCharacterResponse(rianAI)
        time.sleep(10)
        chatController.getCharacterResponse(neonAI)
        time.sleep(10)
        chatController.getCharacterResponse(ranAI)
        time.sleep(10)
        chatController.getCharacterResponse(ruminesAI)
        time.sleep(10)


        for i in range(8):
            try:
                conversationTimingChecker.check_conversation_timing_with_delay(
                    fifoPlayer.get_file_queue_length)
                
                if i % 8 == 0:
                    chatController.addContextAll(
                        'system', "[場面説明]\n別の話題を提案して、話してください。")
                    # chatController.addContextAll(
                    #     'system', "さらにボカロ関連の別の話題を提案して、話してください。")
                elif i % 4 == 0:
                    chatController.addContextAll(
                        'system', "\n話の深堀をして話してください。")
                
                

            except openai.error.RateLimitError:
                print("rate limit error")
                time.sleep(5)
            except openai.error.APIError:
                print("API error")
                time.sleep(5)


        # chatController.postCharacterChat(ruminesAI, "こんにちは、みなさん。ルミネスです。")
        # chatController.postCharacterChat(rianAI, "こんにちは、みなさん。リアンです。")
        # chatController.postCharacterChat(ranAI, "こんにちは、みなさん。ランです。")
        # chatController.postCharacterChat(neonAI, "こんにちは、みなさん。ネオンです。")
        
        
        chatController.postCharacterChat(ranAI, "自己紹介も一通り終わったので、そろそろ、コメント欄の方々と交流してみましょうか！みなさん、コメントたくさんしてくださるとうれしいです。")
        

        for i in range(40):
            print("i: ", i)
            try:
                conversationTimingChecker.check_conversation_timing_with_delay(
                    fifoPlayer.get_file_queue_length)
                
                latestChat = youtubeChat.getSelectedChat()
                if latestChat is not None:
                    latestMsg = latestChat["msg"]
                    print(latestMsg)
                    for i in range(10):
                        try:
                            chatController.addComment( 'system', f"{latestMsg}")
                            break
                        except:
                            time.sleep(1)
                
                chatController.getNextCharacterResponse()
                if random.random() < 0.5:
                    chatController.getNextCharacterResponse()
                if random.random() < 0.5:
                    chatController.getNextCharacterResponse()
                # time.sleep(10)

                if i % 8 == 0:
                    chatController.addContextAll(
                        'system', "[場面説明]\n別の話題を提案して、話してください。")
                    # chatController.addContextAll(
                    #     'system', "さらにボカロ関連の別の話題を提案して、話してください。")
                elif i % 4 == 0:
                    chatController.addContextAll(
                        'system', "\n話の深堀をして話してください。")


            except openai.error.RateLimitError:
                print("rate limit error")
                time.sleep(5)
            except openai.error.APIError:
                print("API error")
                time.sleep(5)
        

        chatController.addContextAll('system', "[場面説明]配信終了まであと3分です。配信の感想を語ってください。")
        
        chatController.getCharacterResponse(rianAI)
        chatController.getCharacterResponse(ranAI)
        chatController.getCharacterResponse(neonAI)
        chatController.getCharacterResponse(ruminesAI)
        
        chatController.addContextAll('system', "[場面説明]デビュー配信を楽しみにしている視聴者に感謝したあと、チャンネル登録するように促してください。")
        chatController.getCharacterResponse(ranAI)
    
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
