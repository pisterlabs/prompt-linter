from llm.chatbot_with_history import ChatBotWithHistory
from langchain.llms import VertexAI
from langchain.llms import OpenAI
from controller import Controller
from statemachine.statemachine import CaseStateMachine
from dotenv import load_dotenv
import logging
import asyncio
import websockets


async def handler(websocket, path):
    async def on_input():
        await websocket.send("input_request")
        input = await websocket.recv()
        return input

    async def on_output(output: str):
        # filter whether to send the output
        production_tags = ["Interviewer", "Candidate"]
        for tag in production_tags:
            if output.startswith(tag):
                return await websocket.send(output)
                

    print("New Connection established")
    llm = VertexAI(max_output_tokens=2048)
    # llm = OpenAI(
    #     model_name='gpt-3.5-turbo',
    # )

    case_state_machine = CaseStateMachine("cases/case.json")
    chatbot = ChatBotWithHistory(llm=llm)

    controller = Controller(chatbot=chatbot, state_machine=case_state_machine, on_input=on_input, on_output=on_output)
    await controller.run_complete_case()


async def main():
    load_dotenv()

    # setup logger
    logname = 'logs/casey.log'
    logging.basicConfig(filename=logname,
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

    logging.info("Running Casey Casebot")

    logger = logging.getLogger('casey')

    async with websockets.serve(handler, "", 8001):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())