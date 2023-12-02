import json
from django.conf import settings
from ai_apps import forms
from utils.base import BaseResponse
from django.shortcuts import render
import openai
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse


def customer_service_assistant(request):
    # 只是为了页面展示
    CustomerServiceAssistantForm = forms.CustomerServiceAssistantForm()

    return render(request, "customer_service_assistant.html",
                  {"CustomerServiceAssistantForm": CustomerServiceAssistantForm})


# def translate(request):
#
#     res = BaseResponse()
#     # print(request.POST)
#
#     text = request.POST.get('text')
#     # print(type(text))
#     # template里面的变量要注意空格
#     try:
#         template = """
#         translate English to Chinese:
#         English: {original_text}
#         Chinese:
#         """
#         prompt = PromptTemplate(template=template, input_variables=["original_text"])
#         llm = OpenAI(streaming=True,verbose=True, temperature=0)# 不写参数就是默认的，默认model_name = "text-davinci-003"
#         llm_chain = LLMChain(prompt=prompt, llm=llm,callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
#         original_text = text
#         result = llm_chain.run(original_text)
#         res.status = True
#         # res.date的值是一个字典，添加一个键值对，key是translation，value是result
#         res.data = {}
#         res.data["translation"] = result
#         print(result)
#     except Exception as e:
#         print(e)
#
#     return JsonResponse(res.dict, json_dumps_params={'ensure_ascii': False})

# 原生openai,使用completion，贵，不推荐

# def translate(request):
#
#     res = BaseResponse()
#     # print(request.POST)
#
#     text = request.POST.get('text')
#     # print(type(text))
#     # template里面的变量要注意空格
#     try:
#         template = """
#         translate English to Chinese:
#         English: {original_text}
#         Chinese:
#         """
#         prompt=template.format(original_text=text)
#         completion = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0)
#         result = completion.choices[0].text
#         res.status = True
#         # res.date的值是一个字典，添加一个键值对，key是translation，value是result
#         res.data = {}
#         res.data["translation"] = result
#         #print(completion)
#         #print(result)
#     except Exception as e:
#         print(e)
#
#     return JsonResponse(res.dict, json_dumps_params={'ensure_ascii': False})

# 原生openai,使用chatcompletion，便宜，推荐


def pack_reply_stream(request):
    # 当请求头stop为true时，停止推送数据
    stop = request.GET.get('stop')

    if stop == 'false':
        received_email = request.GET.get('customer_message')
        include_info = request.GET.get('include_info')
        my_extra_requirement = request.GET.get('my_extra_requirement')
        stream_data = generate_stream_data(received_email, include_info, my_extra_requirement)
        response = StreamingHttpResponse(stream_data, status=200, content_type='text/event-stream')
        # 针对浏览器端--浏览器不应该缓存响应内容。这样可以确保每次请求都会从服务器获取最新的数据，而不是使用本地缓存。
        response['Cache-Control'] = 'no-cache'
        # 针对nginx服务器端--服务器端不缓存数据，nginx就不会缓存数据，这样就可以实时看到数据了
        response['X-Accel-Buffering'] = 'no'
        return response
    return HttpResponse('后台已经停止推送数据')


def generate_stream_data(customer_message, include_info, my_extra_requirement):
    template = """
                We received a message from a customer ：{customer_message},
                ##
                help me write a reply in English that includes this information: {include_info}? 
                ##
                make sure the replay in a casual chat format without a letter structure. Feel free to use emojis if necessary.
                ##
                Additionally, when giving the reply, follow these conditions: {my_extra_requirement}.
                """
    prompt = template.format(
        customer_message=customer_message,
        include_info=include_info,
        my_extra_requirement=my_extra_requirement
    )
    chunks = openai.ChatCompletion.create(
        model=settings.MODELTYPES.get("gpt4"),
        messages=[{"role": "system",
                   "content": "You are a reliable AI live chat customer service assistant who can help you write satisfying English messages based on your prompts."},
                  {"role": "user", "content": prompt}
                  ],
        temperature=0,
        stream=True,
    )
    for chunk in chunks:

        result = chunk.choices[0].get("delta", {}).get("content")
        finish_reason = chunk.choices[0].get("finish_reason")
        if result is None:
            continue
        if result is not None:
            json_str = {"content": result}
            bytes_str = json.dumps(json_str, ensure_ascii=False)
            yield f"data: {bytes_str}\n\n".encode('utf-8')
        if finish_reason == "stop":
            break
    yield 'data: \n\n'
