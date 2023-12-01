import asyncio
import json
import os
from guidance.guidance import GuidanceController
from guidance.buffer import RingBuffer
from aiohttp import web, WSMsgType

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


async def testhandle(request):
    return web.Response(text='Welcome to MyStetho for Pets')

# SOCKETS
async def guidance_ws_handler(request):
    print('Websocket connection - starting')
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print('Websocket connection - ready')

    guidanceController = None
    audioBuffer = None
    fs = None

    async for msg in ws:

        if msg.type == WSMsgType.BINARY:
            print('Websocket connection - received binary')
            audioBuffer.append_list(bin2int16(msg.data))

            if audioBuffer.isFull():
                try:
                    prediction = guidanceController.predict_item(audioBuffer.data, fs)
                    print("prediction", prediction.tolist())
                    await ws.send_str(json.dumps({'dataType': 'prediction', 'data': prediction.tolist()}))

                except Exception as inst:
                    print(type(inst))  # the exception instance
                    print(inst.args)  # arguments stored in .args
                    print(inst)

            else:
                spectrogramawait ws.send_str(json.dumps({'dataType': 'message', 'data': 'Audio buffer loading'}))

        if msg.type == WSMsgType.TEXT:
            print('Websocket connection - received text')
            message = json.loads(msg.data)

            if message['action'] == 'load model':
                fs = message['fs']
                audioBuffer = RingBuffer(int((float(fs)/4000.0)*4250))

                try:
                    guidanceController = GuidanceController()
                    guidanceController.load_model()
                    print(guidanceController)
                    await ws.send_str(json.dumps({'dataType': 'message', 'data': 'AI model loaded'}))

                except Exception as inst:
                    print(type(inst))  # the exception instance
                    print(inst.args)  # arguments stored in .args
                    print(inst)

            elif message['action'] == 'release model':
                guidanceController.release_model()
                await ws.send_str(json.dumps({'dataType': 'message', 'data': 'AI model released'}))
                await ws.close()

            elif message['action'] == 'predict':
                audioBuffer.append_list(message['data'])

                if audioBuffer.isFull():
                    try:
                        prediction = guidanceController.predict_item(audioBuffer.data, fs)
                        print("prediction", prediction.tolist())
                        await ws.send_str(json.dumps({'dataType': 'prediction', 'data': prediction.tolist()}))

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
    app.on_shutdown.append(on_shutdown)
    app.router.add_route('GET', '/', testhandle)
    app.router.add_route('GET', '/guidance_ws', guidance_ws_handler)
    web.run_app(app, port=8080)

# # SERVER
#app = web.Application()
#app.router.add_route('GET', '/', testhandle)
#app.router.add_route('GET', '/guidance_ws', guidance_ws_handler)
