"""
このモジュールは、OpenAI GPT-3.5の出力とVtubeStudio APIを統合して、
Live2Dモデルの動作を制御するためのものです。特定の感情に基づいて
Live2Dの表情を変化させます。

サンプルコード：
    #VtubeStudio のkey認証
    vtsapi=VtubeStudio_API.VTS_main()
    asyncio.get_event_loop().run_until_complete(vtsapi.Authentication_Check())
    # 表現したい感情のリストを設定します
    emotions = ['happy', 'sad']

    async def animate_model():
        emotion_list = []
        for emotion in emotions:
            # それぞれの感情に対するLive2Dパラメータを生成します。
            emotion_dict = VtubeStudio_motion(OpenAI_key, emotion)
            if emotion_dict:
                emotion_list.append(emotion_dict)
        
        for emotion_data in emotion_list:
            # 感情リストに基づいてLive2Dモデルのアニメーションを行います。
            pprint(emotion_data)
            animation_result = await VtubeStudio_emotion(emotion_data)
            if animation_result:
                print("Animation successful.")
            else:
                print("Animation failed.")
        else:
            print("No emotions could be generated.")

    # 非同期ループを開始し、モデルをアニメーション化します。
    loop = asyncio.get_event_loop()
    loop.run_until_complete(animate_model())
"""

import websockets
import openai
import time
import re
import json
import Module.VtubeStudio_API as VtubeStudio_API
import asyncio
import os
from pprint import pprint
from Module.print_color import print_color

def VtubeStudio_GPTmotion(OpenAI_key,emotion,AI_model = 'gpt-3.5-turbo'):
    """
    この関数は、OpenAIのGPT-3.5モデルを利用して、指定した感情に基づくLive2Dパラメータを生成します。
    生成したパラメータはVtubeStudio_emotionに提供することで、Live2Dモデルの動きに変換することができます。
    この関数の応答には10秒から30秒ほど時間がかかります。キャラクターをアニメーションさせる場合は、先にすべてのモーションを保存しておくことをお勧めします。

    引数:
    OpenAI_key -- OpenAIのAPIキー [String]
    emotion -- 表現したい感情 [String]
    AI_model -- OpenAI社が提供しているモデルを設定 [String]

    戻り値:
    dictionary -- VtubeStudio_APIに渡すためのLive2Dパラメータの辞書を返します。Live2Dのパラメータ辞書はVtubeStudio_emotionのemotion_listに用いることができます。取得出来なかった場合はFalseを返します。 [Dictionary]
    """
    openai.api_key=OpenAI_key
    Prompt_template = """Set Live2D model parameters based on the "{0}" emotion. 

1. Summarize "{0}" face and body posture in two sentences.
2. Convert that into Live2D JSON parameters:

- Face Left/Right Rotation (-30 right, 30 left)
- Face Up/Down Rotation (-30 down, 30 up)
- Face Lean Rotation (-30 head right, 30 head left)
- Eye X (-1 pupils right, 1 pupils left, default 0)
- Eye Y (-1 pupils down, 1 pupils up, default 0)
- Mouth Smile (-1 O-shaped, 1 open with gums, default 0)
- Mouth Open (0 closed, 1 open)
- Brow Height Left (-1 near eye (worried), 1 away eye (surprised), default 0.5)
- Brow Height Right (-1 near eye (worried), 1 away eye (surprised), default 0.5)

Provide a JSON object like:
{{
  "Face Left/Right Rotation": 0,
  "Face Up/Down Rotation": 10,
  "Face Lean Rotation": -5,
  "Eye X": 0.5,
  "Eye Y": 0.2,
  "Mouth Smile": 1,
  "Mouth Open": 0.5,
  "Brow Height Left": 0.5,
  "Brow Height Right": 0.5
}}"""

    Prompt = Prompt_template.format(emotion)
    #print_color(f"Prompt:{Prompt}","green")
    #GPT3.5にて問い合わせ
    print_color(f'{AI_model} 問い合わせ開始 ','yellow',True)
    GPT_time= time.time()
    for _ in range(3): #最大3回リトライ
        try:
            responce=openai.ChatCompletion.create(
                model=AI_model,
                messages=[
                    {"role": "system", "content": Prompt}
                ]
            )
            full_result = responce
            result= responce["choices"][0]["message"]["content"]
        except openai.error.RateLimitError:
            print_color("OpenAI RateLimitError:そのモデルは現在、他のリクエストで過負荷になっています。2秒後に再問い合わせを行います。")
            time.sleep(2)
        except openai.error.OpenAIError as e: # OpenAIErrorを追加して、特定のエラーをキャッチ
            if e.type == "server_error":
                print_color("OpenAI Server Error:あなたのリクエストを処理する際に、サーバーにエラーが発生しました。申し訳ありません！エラーが解決しない場合は、リクエストを再試行するか、ヘルプセンター（help.openai.com）にてお問い合わせください。5秒後に再試行します。")
                time.sleep(5)
            else:
                print_color(f"OpenAI 予期せぬエラー: {e}")
                return False
        except Exception as e:
            print_color(f"OpenAI 予期せぬエラー: {e}")
            return False
    #GPTの結果を取得
    print_color(f"表情変更:{emotion} / {AI_model} 問い合わせ時間: {(time.time()-GPT_time):.1f}sec","yellow")
    
    #print(result)
    #Json部分のみを切り取りして、辞書配列に変換
    match = re.search(r'\{(.|\n)+\}', result)
    if match is not None:
        json_string = match.group()
        dictionary = json.loads(json_string)
    else:
        # 一致が見つからなかった場合のエラー処理を行う
        print_color("エラー: モーションについて問い合わせを行ったがモーションが正しく取得できなかった。")
        print_color(f"{responce}")
        return False
    #print(f"表情{emotion}:{result}")
    #辞書配列名をVTSAPIに適合したものに変更
    #{'Face Left/Right Rotation': 0, 'Face Up/Down Rotation': 10, 'Face Lean Rotation': 0, 'Eye X': 0, 'Eye Y': 0, 'Mouth Smile': 1, 'Mouth Open': 0, 'Brow Height': 0.5, 'Brow Height Right': 0.5}
    dictionary['FaceAngleX']=dictionary.pop('Face Left/Right Rotation')
    dictionary['FaceAngleY']=dictionary.pop('Face Up/Down Rotation')
    dictionary['FaceAngleZ']=dictionary.pop('Face Lean Rotation')
    dictionary['EyeLeftX']=dictionary.pop('Eye X')
    dictionary['EyeLeftY']=dictionary.pop('Eye Y')
    dictionary['EyeRightX']=dictionary['EyeLeftX']
    dictionary['EyeRightY']=dictionary['EyeLeftY']
    dictionary['MouthSmile']=dictionary.pop('Mouth Smile')
    dictionary['MouthOpen']=dictionary.pop('Mouth Open')
    dictionary['BrowLeftY']=dictionary.pop('Brow Height Left')
    dictionary['BrowRightY']=dictionary.pop('Brow Height Right')

    #print('GPTで生成された表情パラメータ')
    return dictionary


def VtubeStudio_GPTmotion_Stream(OpenAI_key,emotion,AI_model = 'gpt-3.5-turbo',max_retries=3):
    """
    この関数は、OpenAIのGPT-3.5モデルを利用して、指定した感情に基づくLive2Dパラメータを生成します。
    生成したパラメータはVtubeStudio_emotionに提供することで、Live2Dモデルの動きに変換することができます。
    この関数の応答には10秒から30秒ほど時間がかかります。キャラクターをアニメーションさせる場合は、先にすべてのモーションを保存しておくことをお勧めします。

    引数:
    OpenAI_key -- OpenAIのAPIキー [String]
    emotion -- 表現したい感情 [String]
    AI_model -- OpenAI社が提供しているモデルを設定 [String]

    戻り値:
    dictionary -- VtubeStudio_APIに渡すためのLive2Dパラメータの辞書を返します。Live2Dのパラメータ辞書はVtubeStudio_emotionのemotion_listに用いることができます。取得出来なかった場合はFalseを返します。 [Dictionary]
    """
    openai.api_key=OpenAI_key
    Prompt_template = """Set Live2D model parameters based on the "{0}" emotion. 

1. Summarize "{0}" face and body posture in two sentences.
2. Convert that into Live2D JSON parameters:

- Face Left/Right Rotation (-30 right, 30 left)
- Face Up/Down Rotation (-30 down, 30 up)
- Face Lean Rotation (-30 head right, 30 head left)
- Eye X (-1 pupils right, 1 pupils left, default 0)
- Eye Y (-1 pupils down, 1 pupils up, default 0)
- Mouth Smile (-1 O-shaped, 1 open with gums, default 0)
- Mouth Open (0 closed, 1 open)
- Brow Height Left (-1 near eye (worried), 1 away eye (surprised), default 0.5)
- Brow Height Right (-1 near eye (worried), 1 away eye (surprised), default 0.5)

Provide a JSON object like:
{{
  "Face Left/Right Rotation": 0,
  "Face Up/Down Rotation": 10,
  "Face Lean Rotation": -5,
  "Eye X": 0.5,
  "Eye Y": 0.2,
  "Mouth Smile": 1,
  "Mouth Open": 0.5,
  "Brow Height Left": 0.5,
  "Brow Height Right": 0.5
}}"""

    Prompt = Prompt_template.format(emotion)
    result = ""
    #print_color(f"Prompt:{Prompt}","green")

    #GPT3.5にて問い合わせ
    #print_color(f'{AI_model} Stream VTS問い合わせ開始 ','yellow',True)
    GPT_time= time.time()

    #システムプロンプトの追加
    ChatGPT_Prompt=[{"role": "system", "content": Prompt}]

    for i in range(max_retries):
        try:
            for chunk in openai.ChatCompletion.create(
                model=AI_model,
                messages=ChatGPT_Prompt,
                stream=True,
            ):
                content = chunk["choices"][0].get("delta", {}).get("content")
                #print_color(f"{content}",'green',True)
                if content is not None:
                    result += content
            # 成功したらループを抜ける
            break
        except openai.error.RateLimitError:
            # エラーメッセージを表示
            print_color(f"モデルが他のリクエストで過負荷となっています。")
            # エラーが持続する場合はOpenAIのヘルプセンターに連絡を促すメッセージを表示
            print_color(f"エラーが持続する場合は、help.openai.com に連絡してください。")
            # リトライの間隔を設定（ここでは例として1秒）
            time.sleep(1)
        except Exception as e:
            # その他のエラーを表示
            print_color(f"OpenAI問い合わせ中に、予期せぬエラーが発生しました：{e}")
            break
    
    #GPTの結果を取得
    print_color(f"[表情変更]{AI_model} 問い合わせ時間: {(time.time()-GPT_time):.1f}sec","yellow")
    
    #print(result)
    #Json部分のみを切り取りして、辞書配列に変換
    match = re.search(r'\{(.|\n)+\}', result)
    if match is not None:
        json_string = match.group()
        dictionary = json.loads(json_string)
    else:
        # 一致が見つからなかった場合のエラー処理を行う
        print_color("エラー: モーションについて問い合わせを行ったがモーションが正しく取得できなかった。")
        print_color(f"{result}")
        return False
    #print(f"表情{emotion}:{result}")
    #辞書配列名をVTSAPIに適合したものに変更
    #{'Face Left/Right Rotation': 0, 'Face Up/Down Rotation': 10, 'Face Lean Rotation': 0, 'Eye X': 0, 'Eye Y': 0, 'Mouth Smile': 1, 'Mouth Open': 0, 'Brow Height': 0.5, 'Brow Height Right': 0.5}
    dictionary['FaceAngleX']=dictionary.pop('Face Left/Right Rotation')
    dictionary['FaceAngleY']=dictionary.pop('Face Up/Down Rotation')
    dictionary['FaceAngleZ']=dictionary.pop('Face Lean Rotation')
    dictionary['EyeLeftX']=dictionary.pop('Eye X')
    dictionary['EyeLeftY']=dictionary.pop('Eye Y')
    dictionary['EyeRightX']=dictionary['EyeLeftX']
    dictionary['EyeRightY']=dictionary['EyeLeftY']
    dictionary['MouthSmile']=dictionary.pop('Mouth Smile')
    dictionary['MouthOpen']=dictionary.pop('Mouth Open')
    dictionary['BrowLeftY']=dictionary.pop('Brow Height Left')
    dictionary['BrowRightY']=dictionary.pop('Brow Height Right')

    #print('GPTで生成された表情パラメータ')
    return dictionary

async def VtubeStudio_emotion(emotion_list,animation_timems=1000):
    """
    この関数は、送られたemotion_listに基づいてLive2Dモデルアニメーションを行います。
    emotion_listから取得した各感情に対応するLive2Dパラメータを利用し、VtubeStudio_APIを
    通じてモデルに適用します。

    引数:
    emotion_list -- Live2Dパラメータを変更する感情のリスト [Dictionary]
    animation_timems -- アニメーションの期間(ミリ秒) [int]

    戻り値:
    True/False -- アニメーションの実行が成功したかどうか [bool]

    注意:
    VtubeStudio_emotionは、async defで宣言されています。
    """

    vts_instance = VtubeStudio_API.VTS_main()
    try:
        async with websockets.connect(vts_instance.uri) as websocket:
            #認証確認
            await vts_instance.authenticate(websocket=websocket,plugin_name=vts_instance.pluginName,plugin_developer=vts_instance.pluginDeveloper,api_ver=vts_instance.api_ver,authentication_token=vts_instance.VTS_token)
            #すべてのパラメータの取得
            all_params = await vts_instance.Get_the_value_for_all_Live2D_parameters_in_the_current_model(websocket)
            #print("取得した現在の状態:")
            #for param in all_params:
                #print(f"{param['name']}/value: {param['value']}")
            #print()
            animation_list=[]

            #VTS APIに問い合わせるデータをまとめた辞書配列を作成する
            #トラッキングパラメータリストより、現在値を、エモーションリストよりターゲット値を取得する
            for param in all_params:
                if param['name'] in list(emotion_list.keys()):
                    if float(param['value'])!=float(emotion_list[param['name']]): #値に変化がない場合は無視する
                        animation_list.append({'name':param['name'],'start_value':param['value'],'target_value':emotion_list[param['name']]})
           # print("アニメーションリクエストする内容:")
            #pprint(animation_list)
            
            #アニメーションの実施
            duration = animation_timems #アニメーション時間(ms)
            for t in range(0, duration + 1,10):#10msごとに更新
                request=[]
                for param in animation_list:
                    request.append({"id":param['name'],"value":vts_instance.easeInOutCubic(t,param['start_value'],round(float(param['target_value'])-float(param['start_value']),1),duration)})
                #print(f"リクエスト内容: {request}")
                
                await vts_instance.Feeding_in_data_for_default_or_custom_parameters(websocket,request)
            return True
            
    except Exception as e:
        print(f"VtubeStudio_emotion Error: WebSocket接続時にエラーが発生しました: {e}")
        return False
    
if __name__ == "__main__":

    # OpenAIのAPIキーを設定します
    OpenAI_key = os.getenv("OPENAI_API_KEY")

    #VtubeStudio のkey認証
    vtsapi=VtubeStudio_API.VTS_main()
    asyncio.get_event_loop().run_until_complete(vtsapi.Authentication_Check())
    # 表現したい感情のリストを設定します
    emotions = ['happy', 'sad']

    async def animate_model():
        emotion_list = []
        for emotion in emotions:
            # それぞれの感情に対するLive2Dパラメータを生成します。
            emotion_dict = VtubeStudio_GPTmotion_Stream(OpenAI_key, emotion)
            if emotion_dict:
                emotion_list.append(emotion_dict)
        
        for emotion_data in emotion_list:
            # 感情リストに基づいてLive2Dモデルのアニメーションを行います。
            # pprint(emotion_data)
            animation_result = await VtubeStudio_emotion(emotion_data)
            if animation_result:
                print("Animation successful.")
            else:
                print("Animation failed.")
        else:
            print("No emotions could be generated.")

    # 非同期ループを開始し、モデルをアニメーション化します。
    loop = asyncio.get_event_loop()
    loop.run_until_complete(animate_model())