import asyncio
import json
import os
from guidance.guidance import GuidanceController
from guidance.buffer import RingBuffer
from aiohttp import web, WSMsgType
from PIL import Image
import io
import matplotlib.pyplot as plt
import base64

ROOT = os.path.dirname(__file__)

# USEFULL
def bin2int16(fragment):
    if len(fragment) % 2 != 0:
        fragment = fragment[:-1]

    lenfrag = len(fragment)
    assert (lenfrag % 2 == 0)
    halffrag = int(lenfrag / 2)

    final_frag = []
    for i in range(halffrag):
        local_frag = fragment[i * 2:i * 2 + 2]
        val = int.from_bytes(local_frag, byteorder='little', signed=True)
        final_frag.append(val)
    return final_frag

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

async def testhandle(request):
    return web.Response(text='Welcome to MyStetho spectrogram')

# SOCKETS
async def spectrogram_handler(request):
    print('Websocket connection - starting')
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print('Websocket connection - ready')

    fs = None
    guidanceController = GuidanceController()

    async for msg in ws:

        if msg.type == WSMsgType.BINARY:
            print('Websocket connection - received binary')

            try:
                spectrogram = guidanceController.show_spectrogram(bin2int16(msg.data), fs)
                print("spectrogram", np.size(spectrogram))
                await ws.send_str(json.dumps({'dataType': 'spectrogram', 'data': spectrogram.tolist()}))

            except Exception as inst:
                print(type(inst))  # the exception instance
                print(inst.args)  # arguments stored in .args
                print(inst)

        if msg.type == WSMsgType.TEXT:
            print('Websocket connection - received text')
            message = json.loads(msg.data)

            if message['action'] == 'load fs':
                fs = message['fs']

                try:
                    await ws.send_str(json.dumps({'dataType': 'message', 'data': 'fs loaded'}))

                except Exception as inst:
                    print(type(inst))  # the exception instance
                    print(inst.args)  # arguments stored in .args
                    print(inst)

            elif message['action'] == 'spectrogram':

                spectrogram = guidanceController.spectrogram(message['data'], fs)

                plt.imshow(spectrogram)
                plt.xticks([])
                plt.yticks([])
                fig = plt.gcf()

                img = fig2img(fig)

                in_mem_file = io.BytesIO()
                img.save(in_mem_file, format = "PNG")
                # reset file pointer to start
                in_mem_file.seek(0)
                img_bytes = in_mem_file.read()

                base64_encoded_result_bytes = base64.b64encode(img_bytes)
                base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')

                try:

                    await ws.send_str(json.dumps({'dataType': 'spectrogram_img', 'data': base64_encoded_result_str}))

                except Exception as inst:
                    print(type(inst))  # the exception instance
                    print(inst.args)  # arguments stored in .args
                    print(inst)

                else:
                    await ws.send_str(json.dumps({'dataType': 'message', 'data': 'Audio buffer loading'}))

            else:
                await ws.send_str(json.dumps({'dataType': 'message', 'data': msg.data + '/answer'}))

    print('Websocket connection closed')
    return ws


# LOCAL
if __name__ == '__main__':
    app = web.Application()
    app.router.add_route('GET', '/', testhandle)
    app.router.add_route('GET', '/spectrogram_ws', spectrogram_ws_handler)
    app.router.add_route('POST', '/spectrogram_post', spectrogram_post_handler)
    web.run_app(app, port=8080)

# SERVER
#app = web.Application()
#app.router.add_route('GET', '/', testhandle)
#app.router.add_route('GET', '/spectrogram_ws', spectrogram_handler)
