
class Chat():

    @classmethod
    def Chat0_0_1(self, functionInfo):
        import os
        import json
        from dotenv import load_dotenv
        from datetime import datetime
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        from package.common.common.message.TelegramCtrl import TelegramCtrl
        telegramCtrl = TelegramCtrl(env="env/telegram.env")
        messageJsonStr = telegramCtrl.findMessage()
        messageJson = json.loads(messageJsonStr)
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        updateIdSQL ="""
            select 
                MAX(resultjson::json ->> 'UpdateID') as UpdateID 
            from opsmanagement.opsdetail  
            where 1 = 1 
                and exefunction = 'Chat0_0_1'
                and state = 'FINISH'
        """
        updateIdDF = postgresCtrl.searchSQL(updateIdSQL)
        updateId = int(updateIdDF['updateid'][0])
        for messageDict in messageJson['result']:
            if messageDict['update_id'] <= updateId:
                continue
            if str(messageDict['message']['from']['id']) not in os.getenv("BOT_ALLOW_USERID"):
                continue
            resultObject = {
                "UpdateID": messageDict['update_id'],
                "DateTime": datetime.fromtimestamp(int(messageDict['message']['date'])).strftime("%Y-%m-%d %H:%M:%S"),
                "ChatID": messageDict['message']['chat']['id'],
                "ToID": os.getenv("TELEGRAM_BOT_USERID"),
                "FromID": messageDict['message']['from']['id'],
                "MessageID": messageDict['message']['message_id'],
                "MessageText": messageDict['message']['text'],
            }
            globalObjectDict = {}
            return resultObject, globalObjectDict
        return {}, {}

    @classmethod
    def Chat0_0_2(self, functionInfo):
        import os , time
        import json
        from dotenv import load_dotenv
        from package.common.common.message.TelegramCtrl import TelegramCtrl
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        load_dotenv(dotenv_path="env/postgresql.env")
        telegramCtrl = TelegramCtrl(env="env/telegram.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        updateIdSQL = """
            with Chat0_0_1 as ( 
                select 
                    resultjson::json ->> 'UpdateID' as UpdateID
                    , AA.*
                from opsmanagement.opsdetail AA  
                where 1 = 1 
                    and exefunction = 'Chat0_0_1'
                    and state = 'FINISH'
            ) , Chat0_0_2 as ( 
                select 
                    resultjson::json ->> 'UpdateID' as UpdateID
                from opsmanagement.opsdetail  
                where 1 = 1 
                    and exefunction = 'Chat0_0_2'
                    and state = 'FINISH'
            ) select 
                AA.resultjson
            from Chat0_0_1 AA  
            LEFT join Chat0_0_2 BB on 1 = 1
                and AA.UpdateID = BB.UpdateID
            where 1 = 1 
                and AA.UpdateID is not null
                and BB.UpdateID is null
            order by 
           		AA.UpdateID ASC
        """

        updateDF = postgresCtrl.searchSQL(updateIdSQL)

        for resultJsonStr in updateDF['resultjson'] :
            resultJson = json.loads(resultJsonStr)
            if '@SD' in resultJson["MessageText"]:
                import torch
                from diffusers import StableDiffusionPipeline
                messages = resultJson["MessageText"].replace('@SD', '')
                telegramCtrl.sendMessage(chatID=resultJson["FromID"], message="準備產生相關圖像，請稍後")
                # ======================================== 產生圖片集 ========================================
                pipe = StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path="UnitTest/StableDiffusion/file/model/orangechillmix_v70",  # 模型路徑
                    # low_cpu_mem_usage = True,
                    torch_dtype=torch.float16,  # torch 數據類型
                )
                pipe = pipe.to("cuda")

                generator = torch.Generator().manual_seed(0)

                prompt = resultJson["MessageText"]

                images = pipe(
                    prompt=prompt,  # 正向提示詞
                    # negative_prompt=negativePrompt,           # 負向提示詞 目前是沒啥用處
                    height=512,  # 圖片高度
                    width=512,  # 圖片寬度
                    num_inference_steps=50,  # 推理步數
                    guidance_scale=10,  # 指引縮放 一般來說是7.5 到 12.5之間 提示的權重 小(圖片模糊->圖片合適->過度繪製)大
                    # generator=generator  # 生成器
                ).images

                for image in images:
                    time.sleep(0.1)
                    fileName = "{}.png".format(str(time.time_ns()))
                    filePathName = "Example/P72Telegram/file/out/{}".format(fileName)
                    image.save(filePathName)
                    telegramCtrl.sendPhoto(chatID=resultJson["FromID"], photo=filePathName)

                telegramCtrl.sendMessage(chatID=resultJson["FromID"], message="已產生SD相關圖像")
                # ======================================== 回存訊息到本地端 ========================================
                globalObjectDict = {}
                globalObjectDict["ResponseImage"] = []
                return resultJson, {}
            elif '@GPT' in resultJson["MessageText"] :
                import openai
                from dotenv import load_dotenv
                # ======================================== 產生AI的對話 ========================================
                messages = resultJson["MessageText"].replace('@GPT', '')
                load_dotenv(dotenv_path="env/openai.env")
                openai.api_key = os.getenv("API_KEY")
                response = openai.Completion.create(
                    model=os.getenv("API_MODEL"),
                    prompt=messages,
                    max_tokens=2048,
                    temperature=0.5
                )
                aiMessages = response['choices'][0]['text']
                # ======================================== 回傳給Telegram ========================================
                telegramCtrl.sendMessage(chatID=resultJson["FromID"], message=aiMessages)
                # ======================================== 回存訊息到本地端 ========================================
                globalObjectDict = {}
                globalObjectDict["ResponseText"] = aiMessages
                return resultJson, globalObjectDict
            else :
                # ======================================== 回傳給Telegram ========================================
                responseText = "沒有相關的@的指令，請確認指令內容 請@GPT或@SD"
                telegramCtrl.sendMessage(chatID=resultJson["FromID"], message=responseText)
                # ======================================== 回存訊息到本地端 ========================================
                globalObjectDict = {}
                globalObjectDict["ResponseText"] = responseText
                return resultJson, globalObjectDict
        return {}, {}
