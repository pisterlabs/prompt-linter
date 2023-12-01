import os

import openai
import xml.etree.ElementTree as ET
import subprocess
import shutil

openai.api_key = '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'

def ask_gpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    # 解析回答
    answer = response.choices[0].message.content
    return answer

def decompilation(apk_name):
    command = "java -jar apktool_2.9.0.jar d " + apk_name + " -o app"
    # 运行命令行指令
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 获取命令行输出结果
    output, error = process.communicate()
    # 解码输出结果和错误信息
    # output = output.decode()
    # error = error.decode()
    # 返回输出结果和错误信息
    return output, error






def analyzeOneApk(apk_name,description):
    try:
        # 反编译apk，输出为app文件夹
        decompilation(apk_name)
        # 读取xml
        tree = ET.parse('app/AndroidManifest.xml')
        root = tree.getroot()
        xml_string = ET.tostring(root, encoding='utf-8').decode('utf-8')

        # # 读取游戏介绍（用于测试，现改为直接作为形式参数输入）
        # with open("description.txt", 'r', encoding='utf-8') as file:
        #     description = file.read()

        # 初始化messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        # # 提问xml文件
        # question = "Analyze which permissions and features are used by the decompiled xml file below (only need permissions and features):\n"+xml_string
        # messages.append({"role": "user", "content": question})
        # answer = ask_gpt(messages)
        # print(answer)
        # # 附加回答信息
        # messages.append({"role": "assistant", "content": answer})
        # print("--------------------------------------------------\n")

        # # 提问游戏介绍
        # question = "Take a look at permissions and features that might be used for this game, as described below:\n"+description
        # messages.append({"role": "user", "content": question})
        # answer = ask_gpt(messages)
        # print(answer)
        # # 附加回答信息
        # messages.append({"role": "assistant", "content": answer})
        # print("--------------------------------------------------\n")

        # 提问哪些不该用
        question = "Based on the xml file and game description, what permissions and features should not be applied for according to the privacy policy:\ndescription:\n" + description +"\nxml file:\n" +xml_string
        print(question)
        messages.append({"role": "user", "content": question})
        answer = ask_gpt(messages)
        print(answer)


        # 删除反编译文件夹
        shutil.rmtree("app")

        return answer,xml_string

    except Exception as e:
        if os.path.exists("app") and os.path.isdir("app"):
            shutil.rmtree("app")
        print(f"Error occurred: {str(e)}")





