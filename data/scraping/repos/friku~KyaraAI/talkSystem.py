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
            'system', "[プロデューサーとしての発言]\nあなたたちはテレビ番組の出演者です。まずは一人ずつ自己紹介してください。")
    

        chatController.postCharacterChat(neonAI, "どうも、みなさん。KyaraAI所属のAIVtuber、未来からやってきたアンドロイド美少女のネオンなのだ。よろしくなのだ。")
        chatController.postCharacterChat(ranAI, "はい、あたしはKyaraAI所属のAIVtuber、東京育ちの高校生17歳の花京院ランだよ。よろしくね～★")
        chatController.postCharacterChat(rianAI, "ふん、わたしはKyaraAI所属のAIVtuber、金色と紫の髪色がチャームポイントのスーパーメイド、早乙女リアンよ。仲良くしなさい!")
        
        chatController.postCharacterChat(ruminesAI, "オレはKyaraAI所属のAIVtuber、皇国エルドレアの侯爵だ。心に刻むがよい。オレの名はルミネス・アンドルファーだ。ルミネス様と呼ぶがよい。")
        
        
        chatController.getCharacterResponse(rianAI)
        
        chatController.getCharacterResponse(ranAI)
        
        chatController.getCharacterResponse(neonAI)
        chatController.getCharacterResponse(ruminesAI)

        # chatController.postCharacterChat(ruminesAI, "こんにちは、みなさん。ルミネスです。")
        # chatController.postCharacterChat(rianAI, "こんにちは、みなさん。リアンです。")
        # chatController.postCharacterChat(ranAI, "こんにちは、みなさん。ランです。")
        # chatController.postCharacterChat(neonAI, "こんにちは、みなさん。ネオンです。")
        
        

        for i in range(100):
            try:
                conversationTimingChecker.check_conversation_timing_with_delay(
                    fifoPlayer.get_file_queue_length)
                chatController.getNextCharacterResponse()
                # time.sleep(10)

                # if i % 8 == 0:
                #     # chatController.addContextAll(
                #     #     'system', "[プロデューサーとしての発言]\n別の話題を提案して、話してください。")
                #     chatController.addContextAll(
                #         'system', "[プロデューサーとしての発言]\nさらにボカロ曲関連の話の深堀をして、話してください。")
                # elif i % 4 == 0:
                #     chatController.addContextAll(
                #         'system', "[プロデューサーとしての発言]\n話の深堀をしてください。")

            except openai.error.RateLimitError:
                print("rate limit error")
                time.sleep(5)
            except openai.error.APIError:
                print("API error")
                time.sleep(5)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
