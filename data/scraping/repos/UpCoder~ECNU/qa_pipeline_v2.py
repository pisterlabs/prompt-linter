import json
import logging
import time

import pyaudio
import math
import numpy as np
from demo.demo_asr.utils import GlobalStatus
import asyncio
from demo.demo_asr.zijie.release_interface import get_client
from demo.demo_tts.online.tts_http_demo import get_online_tts_service
import threading
# def demo_client():
#     import socket
#     socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # AF_INET（TCP/IP – IPv4）协议
#     socket_client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
#     socket_client.bind(('localhost', 8998))
#     socket_client.listen()
#     while True:
#         print('client wait message')
#         messages = socket_client.recv(1024).decode('utf-8')
#         # messages = json.loads(messages)
#         # content = messages.get('order', None)
#         content = messages
#         print(f'receive: {content}')
#         data = json.loads(content)
#         print(f'data: {data}')
#         # .send(messages.encode('utf-8'))
#         socket_client.send('AudioFinish'.encode('utf-8'))
# thread1 = threading.Thread(target=demo_client, args=())
# thread1.start()
# time.sleep(1)
global_status = GlobalStatus('localhost', 8998)
CHUNK = 16000
FORMAT = pyaudio.paInt16
CHANNELS = global_status.audio_channel
RATE = global_status.sample_rate
sec_per_chunk = CHUNK / RATE   # 每个chunk代表的实际录音时长
stop_sec_threshold = 5
think_sec_threshold = global_status.think_sec_threshold
stop_value_threshold = global_status.stop_threshold
max_asr_window = 30
_frames = []
appid = "6747655566"  # 项目的 appid
token = "M_3Swzuc6aTtP90HE6VHQ58NmBdF_6Rl"  # 项目的 token
cluster = "volcengine_streaming_common"  # 请求的集群
audio_format = "raw"  # wav 或者 mp3，根据实际音频格式设置
bits = 16
asr_client = get_client(
    {
        'id': 1
    },
    cluster=cluster,
    appid=appid,
    token=token,
    format=audio_format,
    show_utterances=True,
    channel=CHANNELS,
    bits=bits
)
merged_asr_results = []


def is_stop(frames=None):
    num_chunks = math.ceil(stop_sec_threshold / sec_per_chunk)
    think_num_chunks = math.ceil(think_sec_threshold / sec_per_chunk)
    print(f'num_chunks: {num_chunks} / {len(frames) if frames is not None else len(_frames)},'
          f'think_num_chunks: {think_num_chunks}')
    if frames is None:
        binary_values = _frames[int((-1 * num_chunks)):]
    else:
        binary_values = frames[int((-1 * num_chunks)):]
    if len(binary_values) < num_chunks or len(frames) < think_num_chunks:
        return False
    binary_values = b''.join(binary_values)
    np_values = np.frombuffer(binary_values, dtype=np.int16)
    if np.all(np_values < stop_value_threshold):
        print(f'安静了, {np.max(np_values)}')
        return True
    else:
        print(f'max_value: {np.max(np_values)}')
        return False


def merge_asr(cur_asr_result: str, last_asr_result: str):
    """
    合并两句话
    :param cur_asr_result:
    :param last_asr_result:
    :return:
    """
    location_length = 5    # 置信的长度
    end_position = len(last_asr_result)
    while (end_position - location_length) > 0:
        find_str = last_asr_result[end_position-location_length: end_position]
        end_position -= 1
        idx = cur_asr_result.find(find_str)
        if idx == -1:
            continue
        return last_asr_result[:end_position] + cur_asr_result[idx+len(find_str)-1:]
    return cur_asr_result


def get_asr_result_core(frames=None):
    max_frames = max_asr_window // sec_per_chunk
    if frames is None:
        binary_data = b''.join(_frames[int(-1 * max_frames):])
    else:
        binary_data = b''.join(frames[int(-1 * max_frames):])
    arrive_limit = max_frames <= len(_frames)
    result = asyncio.run(asr_client.execute_raw(binary_data, CHANNELS,
                                                bits, RATE))
    if result['payload_msg']['message'] == 'Success':
        cur_asr_result = result['payload_msg']['result'][0]['text']
        if not arrive_limit:
            # 如果没有到达限制，则不需要和前一个进行合并
            merged_asr_result = cur_asr_result
        else:
            # 如果到达了限制，则需要和前一个ASR的结果进行合并
            merged_asr_result = merge_asr(cur_asr_result, merged_asr_results[-1])
            # self.merged_asr_results.append(merged_asr_result)
        return cur_asr_result, merged_asr_result
    else:
        return '', ''


import openai
openai.api_key = "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"


def generate_response(prompt):
    try:
        engine = 'text-davinci-003'
        # engine = 'text-curie-001'
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=0,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0,
            max_tokens=4000 if engine == 'text-davinci-003' else 2000,
            n=1,
            stop=None,
            timeout=20
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(e)
        return '默认答案'


def pipeline():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    binary_data, duration_s, ori_data = get_online_tts_service('你好，我是小艾，你可以向我提问！', False, False)
    if binary_data is not None:
        global_status.send_msg_client.send_message(json.dumps({
            'length': str(duration_s),
            'data': ori_data
        }))
        # time.sleep(0.1)
        # global_status.send_msg_client.send_message(binary_data, is_binary=True)
    while True:
        cur_frames = []
        print('wait receive msg')
        messages = global_status.send_msg_client.socket_client.recv(1024).decode('utf-8')
        print(f'receive msg: {messages}')
        if messages != 'AudioFinish':
            print(f'receive msg: {messages} not AudioFinish')
            continue
        while True:
            print('start record...')
            data = stream.read(CHUNK)
            _frames.append(data)
            cur_frames.append(data)
            if is_stop(cur_frames):
                break
        cur_asr_result, merged_asr_result = get_asr_result_core(cur_frames)
        merged_asr_results.append(merged_asr_result)
        print(f'asr: {merged_asr_result}')
        answer = generate_response(merged_asr_result)
        print('get answer by gpt finished, start play')
        print(f'GPT-3 Answer: {answer}')
        if answer is None or (isinstance(answer, str) and len(answer) == 0):
            answer = '默认答案'
        binary_data, duration_s, ori_data = get_online_tts_service(answer, False, False)
        global_status.send_msg_client.send_message(json.dumps({
            'length': str(duration_s),
            'data': ori_data
        }))
        # global_status.send_msg_client.send_message(json.dumps({
        #     'duration': duration_s
        # }))
        # time.sleep(0.1)
        # global_status.send_msg_client.send_message(binary_data, is_binary=True)
        # messages = conn.recv(1024).decode('utf-8')


if __name__ == '__main__':
    pipeline()