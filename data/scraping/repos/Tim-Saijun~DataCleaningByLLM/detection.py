from openai import OpenAI
from config import config
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=config["apikey"],
    base_url=config["basePath"],
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "\
You need to help find out errors in given data. The first column of data is number, The second column of data is city, the third column of data is the conuntry which the first column belongs to.\n\
Desired Format: list format with the result, result has only two values: 0 or 1, when you find an error in a line,the result is 1 , otherwise the result is 0 . Do not give any unnecessary imformation, return list only!\n\
Example output: [0,0,0,0,1]\n\
Data:\n\
1,New York,United States\n\
2,London,United Kingdom\n\
3,Paris,France\n\
4,Tokyo,China\n\
5,Sydney,Australia",
        }
    ],
    model="gpt-3.5-turbo",
    temperature=0,
)

print(chat_completion)
print(chat_completion.choices[0].message.content)