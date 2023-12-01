import os
import openai
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True,
                        help='prompt of your next question')

    args = parser.parse_args()
    question = args.question

    prompt = f" \
下列是一个与AI助手之间的对话，这个助手擅长编写Python代码，能够提供关于Python编程方面的有效建议。\n \
\n \
Human: 你好。\n \
AI: 你好，我是一个智能助理，能给你提供关于Python程序开发方面的一些建议。\n \
Human: 请给我一个Python3当中，使用ArgumentParser接收命令行参数的简单示例。\n \
AI: 你可以使用下面的示例：\n \
```\n \
import argparse\n \
\n \
parser = argparse.ArgumentParser()\n \
parser.add_argument('-n', '--name', required=True, help='Name of the user')\n \
parser.add_argument('-a', '--age', required=True, help='Age of the user')\n \
\n \
args = parser.parse_args()\n \
name = args.name\n \
age = args.age\n \
\n \
print('Name: %s, Age: %s' % (name, age))\n \
```\n \
Human: 如何用一行代码启动一个http服务器，请给一个简单的例子.\n \
AI:你可以使用Python模块“http.server”中的HTTPServer类来启动一个HTTP服务器，例如：\n \
```\n \
python -m http.server 8080\n \
```\n \
Human: {question}\n \
"

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )

    answer = response['choices'][0]['text']

    print(f'You asked: {question}')
    print(f'AI answered: {answer}')


if __name__ == '__main__':
    main()
