# import sys
#
# sys.path.append('/opt/python')          # this needs to be done since LWA loses the path to the layers
# sys.path.append('/var/task'  )          # path to lambda function code
#
# def run():
#     from osbot_llms.apis.open_ai.API_Open_AI import API_Open_AI
#     from osbot_utils.utils.Dev               import pprint
#     from fastapi                             import FastAPI, Request
#     from fastapi.responses                   import StreamingResponse
#     from osbot_utils.utils.Misc              import str_to_bytes
#
#     app = FastAPI()
#
#     async def streamer(prompt):
#         messages = [{"role": "user", "content": prompt}]
#         #pprint(messages)
#
#         import openai
#         from openai import ChatCompletion
#         api_instance = API_Open_AI()
#
#         openai.api_key = api_instance.api_key()
#
#         response = ChatCompletion.create(model      = api_instance.model      ,
#                                          messages   = messages                ,
#                                          temperature= api_instance.temperature,
#                                          stream     = api_instance.stream     )
#         for chunk in response:
#             if len(chunk['choices'][0]['delta']) != 0:
#                 new_content = chunk['choices'][0]['delta']['content']
#                 yield new_content + '\n'
#
#         # api_instance = API_Open_AI()
#         # for new_content in api_instance.create(messages):
#         #     yield new_content
#
#     @app.get("/")
#     async def index(request: Request):
#         prompt = request.query_params.get('prompt', 'Hi')
#         return StreamingResponse(streamer(prompt), media_type="text/plain; charset=utf-8")
#
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080)
#
# if __name__ == '__main__':
#     run()                                  # to be triggered from run.sh