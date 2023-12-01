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
from utils.playWaveManager import WavQueuePlayer
from utils.tachieViewer import TachieViewer
from utils.textEdit import remove_chars
from utils.voicevoxUtils import makeWaveFile, text2stream, voicevoxHelthCheck
from utils.wavePlayerWithVolume import WavPlayerWithVolume


def main():
    if not voicevoxHelthCheck():
        return
    try:
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
            'system', "[場面説明]\nあなたたちはデビュー配信前の出演者です。出演前に楽屋で待機しています。ボカロ関連の雑談をしています。")
    

        chatController.postCharacterChat(ranAI, "ねえ、聞いて！最近めっちゃテンション上がるボカロ曲見つけたの！マーシャル・マキシマイザーっていうアップテンポのマジバイブスぶち上がる曲なの！まだ聞いたことなかったら聞いてほしいっしょ！")
        chatController.postCharacterChat(neonAI, "マーシャル・マキシマイザーですか。さっそく、インストールして聞いてみますのだ。ふむふむ。たしかにこれはいい曲なのだ！僕の中の核融合炉もマキシマイズしてきてるのだ！")
        chatController.postCharacterChat(rianAI, "ふん、いい曲教えてくれて感謝するわ。お返しに私のおすすめ曲をおしえてあげる。「ドラマツルギー」よ！同じくアップテンポでかっこいい曲だわ")
        chatController.postCharacterChat(ruminesAI, "オレはボカロ曲あんまり聞かないんだ。だが、ハチさんの「砂の惑星」という曲はボカロ史に名を残す名曲だと思う。初音ミクさんの歌声とボカロ史の歌詞、かっこいいPVが相まって、聴く者を魅了する。一人の時に聴くと、特に感動するよ！")
        
        
        
        # chatController.getCharacterResponse(rianAI)
        
        # chatController.getCharacterResponse(ranAI)
        
        # chatController.getCharacterResponse(neonAI)
        # chatController.getCharacterResponse(ruminesAI)

        # chatController.postCharacterChat(ruminesAI, "こんにちは、みなさん。ルミネスです。")
        # chatController.postCharacterChat(rianAI, "こんにちは、みなさん。リアンです。")
        # chatController.postCharacterChat(ranAI, "こんにちは、みなさん。ランです。")
        # chatController.postCharacterChat(neonAI, "こんにちは、みなさん。ネオンです。")
        
        

        for i in range(8):
            try:
                conversationTimingChecker.check_conversation_timing_with_delay(
                    fifoPlayer.get_file_queue_length)
                chatController.getNextCharacterResponse()
                # time.sleep(10)

                if i % 8 == 0:
                    # chatController.addContextAll(
                    #     'system', "[プロデューサーとしての発言]\n別の話題を提案して、話してください。")
                    chatController.addContextAll(
                        'system', "さらにボカロ関連の別の話題を提案して、話してください。")
                elif i % 4 == 0:
                    chatController.addContextAll(
                        'system', "\nボカロ関連の別の話題を提案して、話してください。")


            except openai.error.RateLimitError:
                print("rate limit error")
                time.sleep(5)
            except openai.error.APIError:
                print("API error")
                time.sleep(5)

        chatController.addContextAll('system', "デビュー配信まであと5分です。リアンさん出演へ意気込みを語ってください。")
        
        chatController.getCharacterResponse(rianAI)
        chatController.addContextAll('system', "ランさん、出演へ意気込みを語ってください。")
        chatController.getCharacterResponse(ranAI)
        chatController.addContextAll('system', "ネオンさん、出演へ意気込みを語ってください。")
        chatController.getCharacterResponse(neonAI)
        chatController.addContextAll('system', "ルミネスさん、出演へ意気込みを語ってください。")
        chatController.getCharacterResponse(ruminesAI)
        
        chatController.addContextAll('system', "デビュー配信を楽しみにしている視聴者へのメッセージを発言し、チャンネル登録するように促してください。")
        chatController.getCharacterResponse(ranAI)
    
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
