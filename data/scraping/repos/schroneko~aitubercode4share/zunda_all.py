import time
import requests
import json
import openai
import os
import simpleaudio as sa

def responseFromOpenAI(message):
    OPENAI_API_KEY=""

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY

    start_sequence = "\nAI:"
    restart_sequence = "\nHuman: "

    response = openai.Completion.create(
    model="text-davinci-003",
    # prompt="What is your name?",
    prompt=message,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )

    # print(response)
    return response["choices"][0]["text"]

dir = os.getcwd()

def speakByZunda(replyText):
    # text = "私の名前はずんだもんです。東北地方の応援マスコットをしています。得意なことはしゃべることです。"
    text = replyText
    # 音声合成クエリの作成
    res1 = requests.post('http://127.0.0.1:50021/audio_query',params = {'text': text, 'speaker': 1})
    # 音声合成データの作成
    res2 = requests.post('http://127.0.0.1:50021/synthesis',params = {'speaker': 1},data=json.dumps(res1.json()))
    # wavデータの生成
    with open('test.wav', mode='wb') as f:
        f.write(res2.content)
    wavDir = os.path.join(dir, "test.wav")
    wave_obj = sa.WaveObject.from_wave_file(wavDir)
    play_obj = wave_obj.play()
    play_obj.wait_done()


YT_API_KEY = 'AIzaSyB6j2I-k92CQSBcra0Um65402Wbak4j2tA'

def get_chat_id(yt_url):
    '''
    https://developers.google.com/youtube/v3/docs/videos/list?hl=ja
    '''
    video_id = yt_url.replace('https://www.youtube.com/watch?v=', '')
    print('video_id : ', video_id)

    url    = 'https://www.googleapis.com/youtube/v3/videos'
    params = {'key': YT_API_KEY, 'id': video_id, 'part': 'liveStreamingDetails'}
    data   = requests.get(url, params=params).json()

    liveStreamingDetails = data['items'][0]['liveStreamingDetails']
    if 'activeLiveChatId' in liveStreamingDetails.keys():
        chat_id = liveStreamingDetails['activeLiveChatId']
        print('get_chat_id done!')
    else:
        chat_id = None
        print('NOT live')

    return chat_id


def get_chat(chat_id, pageToken, log_file):
    '''
    https://developers.google.com/youtube/v3/live/docs/liveChatMessages/list
    '''
    url    = 'https://www.googleapis.com/youtube/v3/liveChat/messages'
    params = {'key': YT_API_KEY, 'liveChatId': chat_id, 'part': 'id,snippet,authorDetails'}
    if type(pageToken) == str:
        params['pageToken'] = pageToken

    data   = requests.get(url, params=params).json()

    try:
        for item in data['items']:
            channelId = item['snippet']['authorChannelId']
            msg       = item['snippet']['displayMessage']

            # 再生するよ！
            replyFromAI = responseFromOpenAI(msg)
            print(replyFromAI)
            speakByZunda(replyFromAI)

            usr       = item['authorDetails']['displayName']
            #supChat   = item['snippet']['superChatDetails']
            #supStic   = item['snippet']['superStickerDetails']
            log_text  = '[by {}  https://www.youtube.com/channel/{}]\n  {}'.format(usr, channelId, msg)
            with open(log_file, 'a') as f:
                print(log_text, file=f)
                print(log_text)
        print('start : ', data['items'][0]['snippet']['publishedAt'])
        print('end   : ', data['items'][-1]['snippet']['publishedAt'])

    except:
        pass

    return data['nextPageToken']

def main(yt_url):
    slp_time        = 10 #sec
    iter_times      = 90 #回
    take_time       = slp_time / 60 * iter_times
    print('{}分後　終了予定'.format(take_time))
    print('work on {}'.format(yt_url))

    log_file = yt_url.replace('https://www.youtube.com/watch?v=', '') + '.txt'
    with open(log_file, 'a') as f:
        print('{} のチャット欄を記録します。'.format(yt_url), file=f)
    chat_id  = get_chat_id(yt_url)

    nextPageToken = None
    for ii in range(iter_times):
        #for jj in [0]:
        try:
            print('\n')
            nextPageToken = get_chat(chat_id, nextPageToken, log_file)
            time.sleep(slp_time)
        except:
            break

if __name__ == '__main__':
    yt_url = input('Input YouTube URL > ')
    main(yt_url)
