import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"


def get_response(prompt, temperature=1.0):
    completions = openai.Completion.create(
        engine=COMPLETION_MODEL,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=temperature,
    )
    message = completions.choices[0].text
    return message


if __name__ == '__main__':
    prompt = '请你用朋友的语气回复给到客户，并称他为“亲”，他的订单已经发货在路上了，预计在3天之内会送达，订单号2021AEDG，我们很抱歉因为天气的原因物流时间比原来长，感谢他选购我们的商品。'
    # 亲，你的订单号2021AEDG已经发货，预计会在3天内送达，很抱歉因为天气原因给您带来不便，感谢你一直以来对我们产品的支持！
    print(get_response(prompt))
