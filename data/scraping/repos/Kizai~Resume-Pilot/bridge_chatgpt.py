import json
import time
import logging
import traceback
import requests
import importlib
from config import API_URL, API_KEY, TIMEOUT_SECONDS, MAX_RETRY, LLM_MODEL

timeout_bot_msg = 'Request timeout. Network error. Please check proxy settings in config.py.' + \
                  'Network error. Please check if the proxy server is available and if the proxy settings are correct. The format must be [protocol]://[address]:[port], and all parts are required.'


def get_full_error(chunk, stream_response):
    """
    Get the complete error message returned from OpenAI.
    """
    while True:
        try:
            chunk += next(stream_response)
        except:
            break
    return chunk


def predict_no_ui(inputs, top_p, temperature, history=None, sys_prompt=""):
    """
    Send to chatGPT, wait for reply, complete in one go, no intermediate process will be displayed.
    A simplified version of the predict function.
    It is used when the payload is relatively large, or to implement multi-line and complex functions with nesting.
    inputs is the input of this query
    top_p, temperature are internal tuning parameters of chatGPT
    history is a list of previous conversations
    (Note that whether it is inputs or history, if the content is too long, it will trigger an error that the number of tokens overflows, and then raise ConnectionAbortedError)
    """
    if history is None:
        history = []
    headers, payload = generate_payload(inputs, top_p, temperature, history, system_prompt=sys_prompt, stream=False)

    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=False
            response = requests.post(API_URL, headers=headers,
                                     json=payload, stream=False, timeout=TIMEOUT_SECONDS * 2)
            break
        except requests.exceptions.ReadTimeout:
            retry += 1
            traceback.print_exc()
            if retry > MAX_RETRY:
                raise TimeoutError
            if MAX_RETRY != 0:
                print(f'Request timed out, retrying ({retry}/{MAX_RETRY}) ……')

    try:
        result = json.loads(response.text)["choices"][0]["message"]["content"]
        return result
    except Exception:
        if "choices" not in response.text:
            print(response.text)
        raise ConnectionAbortedError("Json parsing is irregular, the text may be too long" + response.text)


def predict_no_ui_long_connection(inputs, top_p, temperature, history=None, sys_prompt="", observe_window=None):
    """
    Send to chatGPT, wait for reply, complete in one go, no intermediate process will be displayed. But the method of stream is used internally to avoid the network cable being pinched in the middle.
    inputs:
     is the input for this query
    sys_prompt:
     System silent prompt
    top_p, temperature:
     Internal tuning parameters of chatGPT
    history:
     is a list of previous conversations
    observe_window = None:
     It is responsible for passing the output part across threads. Most of the time, it is only for the fancy visual effect, and it can be left blank. observe_window[0]: observation window. observe_window[1]: watchdog
    """
    if history is None:
        history = []
    watch_dog_patience = 5  # Watchdog's patience, set it to 5 seconds
    headers, payload = generate_payload(inputs, top_p, temperature, history, system_prompt=sys_prompt, stream=True)
    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=False
            response = requests.post(API_URL, headers=headers,
                                     json=payload, stream=True, timeout=TIMEOUT_SECONDS)
            break
        except requests.exceptions.ReadTimeout:
            retry += 1
            traceback.print_exc()
            if retry > MAX_RETRY:
                raise TimeoutError
            if MAX_RETRY != 0:
                print(f'Request timed out, retrying ({retry}/{MAX_RETRY}) ……')

    stream_response = response.iter_lines()
    result = ''
    while True:
        try:
            chunk = next(stream_response).decode()
        except StopIteration:
            break
        if len(chunk) == 0:
            continue
        if not chunk.startswith('data:'):
            error_msg = get_full_error(chunk.encode('utf8'), stream_response).decode()
            if "reduce the length" in error_msg:
                raise ConnectionAbortedError("OpenAI rejected the request:" + error_msg)
            else:
                raise RuntimeError("OpenAI rejected the request：" + error_msg)
        json_data = json.loads(chunk.lstrip('data:'))['choices'][0]
        delta = json_data["delta"]
        if len(delta) == 0:
            break
        if "role" in delta:
            continue
        if "content" in delta:
            result += delta["content"]
            print(delta["content"], end='')
            if observe_window is not None:
                # Observation window to display the acquired data
                if len(observe_window) >= 1:
                    observe_window[0] += delta["content"]
                # Watchdog, if the dog is not fed after the deadline, it will be terminated
                if len(observe_window) >= 2:
                    if (time.time() - observe_window[1]) > watch_dog_patience:
                        raise RuntimeError("program terminated。")
        else:
            raise RuntimeError("Unexpected Json structure：" + delta)
    if json_data['finish_reason'] == 'length':
        raise ConnectionAbortedError("It ends normally, but it shows insufficient Token, resulting in incomplete output. Please reduce the amount of text entered at a time.")
    return result


def predict(inputs, top_p, temperature, chatbot=None, history=None, system_prompt='',
            stream=True, additional_fn=None):
    """
    Send to chatGPT to stream the output.
    Used for basic dialog functionality.
    inputs is the input of this query
    top_p, temperature are internal tuning parameters of chatGPT
    history is a list of previous conversations (note that whether it is inputs or history, if the content is too long, it will trigger an error that the number of tokens overflows)
    chatbot is the dialog list displayed in the WebUI, modify it, and then yeild out, you can directly modify the content of the dialog interface
    additional_fn represents which button is clicked, see functional.py for the button
    """
    if history is None:
        history = []
    if chatbot is None:
        chatbot = []
    if additional_fn is not None:
        import core_functional
        importlib.reload(core_functional)
        core_functional = core_functional.get_core_functions()
        if "PreProcess" in core_functional[additional_fn]:
            inputs = core_functional[additional_fn]["PreProcess"](inputs)
        inputs = core_functional[additional_fn]["Prefix"] + inputs + core_functional[additional_fn]["Suffix"]

    if stream:
        raw_input = inputs
        logging.info(f'[raw_input] {raw_input}')
        chatbot.append((inputs, ""))
        yield chatbot, history, "waiting for response"

    headers, payload = generate_payload(inputs, top_p, temperature, history, system_prompt, stream)
    history.append(inputs)
    history.append(" ")

    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=True
            response = requests.post(API_URL, headers=headers,
                                     json=payload, stream=True, timeout=TIMEOUT_SECONDS)
            break
        except:
            retry += 1
            chatbot[-1] = (chatbot[-1][0], timeout_bot_msg)
            retry_msg = f"，retrying ({retry}/{MAX_RETRY}) ……" if MAX_RETRY > 0 else ""
            yield chatbot, history, "Request timed out" + retry_msg
            if retry > MAX_RETRY:
                raise TimeoutError

    gpt_replying_buffer = ""

    is_head_of_the_stream = True
    if stream:
        stream_response = response.iter_lines()
        while True:
            chunk = next(stream_response)
            # print(chunk.decode()[6:])
            if is_head_of_the_stream:
                # The first frame of the data stream does not carry content
                is_head_of_the_stream = False
                continue

            if chunk:
                try:
                    if len(json.loads(chunk.decode()[6:])['choices'][0]["delta"]) == 0:
                        # Determined as the end of the data stream, gpt_replying_buffer is also written
                        logging.info(f'[response] {gpt_replying_buffer}')
                        break
                    # Handle the body of the stream
                    chunkjson = json.loads(chunk.decode()[6:])
                    status_text = f"finish_reason: {chunkjson['choices'][0]['finish_reason']}"
                    # If an exception is thrown here, it is usually because the text is too long, see the output of get_full_error for details
                    gpt_replying_buffer = gpt_replying_buffer + json.loads(chunk.decode()[6:])['choices'][0]["delta"][
                        "content"]
                    history[-1] = gpt_replying_buffer
                    chatbot[-1] = (history[-2], history[-1])
                    yield chatbot, history, status_text

                except Exception:
                    traceback.print_exc()
                    yield chatbot, history, "Json parsing is irregular"
                    chunk = get_full_error(chunk, stream_response)
                    error_msg = chunk.decode()
                    if "reduce the length" in error_msg:
                        chatbot[-1] = (chatbot[-1][0],
                                       "Reduce the length. This input is too long, or the historical data is too long. The historical cache data is now released, you can try again.")
                        history = []  # 清除历史
                    elif "Incorrect API key" in error_msg:
                        chatbot[-1] = (chatbot[-1][0],
                                       "Incorrect API key. OpenAI denies service on the grounds that an incorrect API_KEY is provided.")
                    elif "exceeded your current quota" in error_msg:
                        chatbot[-1] = (chatbot[-1][0],
                                       "You exceeded your current quota. OpenAI refuses service due to insufficient account quota..")
                    else:
                        from tools.toolbox import regular_txt_to_markdown
                        tb_str = '```\n' + traceback.format_exc() + '```'
                        chatbot[-1] = (chatbot[-1][0],
                                       f"Exception\n\n{tb_str} \n\n{regular_txt_to_markdown(chunk.decode()[4:])}")
                    yield chatbot, history, "Json Exception" + error_msg
                    return


def generate_payload(inputs, top_p, temperature, history, system_prompt, stream):
    """
    Integrate all information, select LLM model, generate http request, prepare for sending request
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    conversation_cnt = len(history) // 2

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_cnt:
        for index in range(0, 2 * conversation_cnt, 2):
            what_i_have_asked = {"role": "user", "content": history[index]}
            what_gpt_answer = {"role": "assistant", "content": history[index + 1]}
            if what_i_have_asked["content"] != "":
                if what_gpt_answer["content"] == "":
                    continue
                if what_gpt_answer["content"] == timeout_bot_msg:
                    continue
                messages.append(what_i_have_asked)
                messages.append(what_gpt_answer)
            else:
                messages[-1]['content'] = what_gpt_answer['content']

    what_i_ask_now = {"role": "user", "content": inputs}
    messages.append(what_i_ask_now)

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,  # 1.0,
        "top_p": top_p,  # 1.0,
        "n": 1,
        "stream": stream,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    print(f" {LLM_MODEL} : {conversation_cnt} : {inputs}")
    return headers, payload
