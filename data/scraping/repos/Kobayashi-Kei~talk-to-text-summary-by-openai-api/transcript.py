import os
import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
import openai
import ffmpeg

openai.api_key_path = 'apikey.txt'

def main():    
    # 入力
    dirPath = ""
    fileName = ""

    # m4aファイルをmp3に変換
    if "m4a" in  fileName:
        m4aToMp3(dirPath, fileName)
        fileName = fileName.replace(".m4a", ".mp3")
        
    # 音声ファイルを分割
    splitVoiceFile(dirPath, fileName)
    
    # 出力先ディレクトリを作成
    if os.path.exists(dirPath + "text") == False:
        os.mkdir(dirPath + "text")
    if os.path.exists(dirPath + "output") == False:
        os.mkdir(dirPath + "output")

    # Whisperで文字起こし
    chunkList = os.listdir(dirPath + "chunk/")
    for chunk in chunkList:
        if os.path.exists(dirPath + "text/" + chunk.replace(".mp3", ".txt")):
            continue
        print("Whisperで文字起こし中:" + chunk)
        audio_chunk = open(dirPath + "chunk/" + chunk, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_chunk)
        txt = transcript['text']
        f = open(dirPath + "text/" + chunk.replace(".mp3", "") + ".txt", "w")
        f.write(txt)
        f.close()
        
    
    # テキストファイルを結合 & ChatGPTでトピック抽出
    txtList = os.listdir(dirPath + "text/")
    for txt_file in txtList:
        f = open(dirPath + "text/" + txt_file, "r")
        txt = f.read()
        f.close()
        f = open(dirPath + "output/output.txt", "a")
        f.write(txt)
        f.close()

        # ChatGPTでトピック抽出
        print("ChatGPTでトピック抽出中:" + txt_file)
        txt_chunks = split_string(txt)
        topic_txt = ""
        for txt_chunk in txt_chunks:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "以下のテキストから主要なトピックを箇条書きで列挙してください．それぞれの項目単体で理解できるように内容を省略せずに書いてください．\n以下がテキストです：\n" + txt_chunk },
                ]   
            )
            topic_txt += response["choices"][0]["message"]["content"]
        
        f = open(dirPath + "text/" + txt_file.replace(".txt", "_topic.txt"), "a")
        f.write(topic_txt)
        f.close()
        f = open(dirPath + "output/output_topic.txt", "a")
        f.write(topic_txt)
        f.close()


        # 以下は効果が薄いのでやらない    
        # # GPTで整形    
        # txt_chunks = split_string(txt)
        # result_txt = ""
        # for txt_chunk in txt_chunks:
        #     response = openai.ChatCompletion.create(
        #         model="gpt-3.5-turbo",
        #         messages=[
        #             {"role": "user", "content": "以下の文章はspeechToTextで変換したものです．誤字脱字をできる限り修正して読みやすい文章にしてください．" + "\n" + txt_chunk },
        #         ]   
        #     )
        #     # print(response["choices"][0]["message"]["content"])
        #     result_txt += response["choices"][0]["message"]["content"]
        
        # f = open(dirPath + "text/" + txt_file.replace(".txt", "_gpt.txt"), "a")
        # f.write(result_txt)
        # f.close()
        
        # # output.txtに文字起こし &　GPTでの整形結果を書き込み
        # f = open(dirPath + "output_gpt.txt", "a")
        # f.write(result_txt)
        # f.close()
    

def m4aToMp3(dirPath, fileName):
    filePath = dirPath + fileName
    
    stream = ffmpeg.input(filePath)
    stream = ffmpeg.output(stream, filePath.replace(".m4a", ".mp3"))
    ffmpeg.run(stream)
    
    # root, ext = os.path.splitext(filePath)
    # newname = '%s.mp3' % root
    # cmd = 'ffmpeg -i %s -sameq %s' % (filePath, newname)
    # print(cmd)
    # subprocess.run(cmd, shell=True)

def splitVoiceFile(dirPath, fileName):
    audio = AudioSegment.from_file(dirPath + fileName, format="mp3")

    # 1000秒(1000000ミリ秒)ごとに分割
    chunks = make_chunks(audio, 1000000)

    # 分割した音声ファイルを出力する
    for i, chunk in enumerate(chunks):
        if os.path.exists(dirPath + "chunk") == False:
            os.mkdir(dirPath + "chunk")
        chunk_name =  dirPath + "chunk/chunk{0}.mp3".format(i)
        print("exporting", chunk_name)
        chunk.export(chunk_name, format="mp3")
        
def split_string(text, chunk_size=1000, overlap=50):
    result = []
    length = len(text)
    start = 0
    end = chunk_size + overlap

    while start < length:
        chunk = text[start:end]
        result.append(chunk)
        start = start + chunk_size
        end = start + chunk_size + overlap

    return result


if __name__ == "__main__":
    main()

