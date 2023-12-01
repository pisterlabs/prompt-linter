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
from utils.YTComment import YTComment


def main():
    if not voicevoxHelthCheck():
        return
    try:
        yt_url = 'https://www.youtube.com/watch?v=vdK6kuqjn10' # youtubeのURL
        youtubeChat = YTComment(yt_url)

        fifoPlayer = FIFOPlayer()
        fifoPlayer.playWithFIFO()


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
            'system', "[場面説明]\nあなたたちはいまディストピア短編小説を考えています。ディストピア短編小説のテーマについてアイディア出しをしてください。")
        # chatController.addContextAll(
        #     'system', "[]\nあなたのワードは「トマト」です。")



        try:
            conversationTimingChecker.check_conversation_timing_with_delay(
                fifoPlayer.get_file_queue_length)


            chatController.getNextCharacterResponse(outputNum = 30)
            chatController.getNextCharacterResponse(outputNum = 30)
            chatController.getNextCharacterResponse(outputNum = 30)


            chatController.addContextAll('system', "[場面説明]\nこれまでに出た情報を統合的に考慮して、段階的かつ論理的に小説のテーマに沿って世界観を考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 200)

            chatController.addContextAll('system', "[場面説明]\nこれまでに出た情報を統合的に考慮して、物語の主人公の名前とプロフィールを段階的かつ論理的に考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 200)

            chatController.addContextAll('system', "[場面説明]\nこれまでに出た情報を統合的に考慮して、ほかの登場人物と役割について考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 100)
            chatController.addContextAll('system', "[場面説明]\nその世界で奇妙なことが起こりますが、奇妙なことが起こる前の物語をこれまで出た全ての情報を総合的に考慮し段階的かつ論理的に考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 200)

            chatController.addContextAll('system', "[場面説明]\nその世界で奇妙なことが起こりました。どんな奇妙なことが起こったのかをこれまで出た全ての情報を総合的に考慮し段階的かつ論理的に考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 200)
            # chatController.getNextCharacterResponse(outputNum = 200)

            chatController.addContextAll('system', "[場面説明]\n最後に読み手がハッと驚くオチを作ります。奇妙なことが起こった衝撃の理由をこれまで出た全ての情報を総合的に考慮し発言してください。")
            chatController.getNextCharacterResponse(outputNum = 200)

            chatController.addContextAll('system', "[場面説明]\nこの物語にタイトルを作ります。タイトルを考える際にはこれまで出た全ての情報を総合的に考慮し発言してください。")
            chatController.getNextCharacterResponse(outputNum = 40)
            chatController.addContextAll('system', "[場面説明]\nこの物語を読んだ感想を述べてください。")
            chatController.getNextCharacterResponse(outputNum = 40)




        except openai.error.RateLimitError:
            print("rate limit error")
            time.sleep(5)
        except openai.error.APIError:
            print("API error")
            time.sleep(5)
        print("end")

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
