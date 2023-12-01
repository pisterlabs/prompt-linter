import openai
from gpt_2_create_question import decorate_query
from gpt_0_basic_info import api_key, csv_file_path
openai.api_key = api_key


def question(input_query):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_query}
        ],
        max_tokens=100,
        n=1,
        temperature=0.5,
    )
    """
    n: 这个参数用于指定你想要生成的回复数量。如果设置为1（默认值），则生成一个回复。如果设置为大于1的整数，将生成多个独立的回复。
    这在某些情况下可能很有用，例如，当你想要从多个回复中选择一个，或者在多个回复之间进行排序。

    temperature: 这个参数控制生成文本时的随机性。它是一个浮点数。较低的值（如0.5）会使生成的文本更加保守、聚焦和一致，而较高的值（如1.0）
    会使生成的文本更加随机和多样。调整temperature参数可以帮助你找到合适的生成文本的多样性和创造性。在这个例子中，
    temperature被设置为0.5，这意味着生成的回复会比较保守和一致。
    """
    output_text = response.choices[0].message['content'].strip()
    return output_text


if __name__ == '__main__':
    query = '亁颐堂是做什么的'
    new_query = decorate_query(query, filepath=csv_file_path)
    print(new_query)
    # 产生如下的问题：
    """
    请使用上下文尽可能真实、自然地回答问题，如果答案未包含在上下文中，请不要编造回答，并且不要在回答中包含”根据上下文”这个短语。
    
    上下文：
    当有人问：公司名称, 请回答：亁颐堂科技有限责任公司
    当有人问：亁颐堂是做什么的, 请回答：亁颐堂是一个网络培训公司
    
     问题: 亁颐堂是做什么的
     回答:？
    """
    # 把上面的问题打给OPEN AI, 就能正确的回答问题
    # 亁颐堂是一个网络培训公司。
    print(question(new_query))
