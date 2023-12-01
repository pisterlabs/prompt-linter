import re
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from dotenv import load_dotenv
from langchain.chains import LLMChain
from openai import OpenAI

# Load .env file
load_dotenv()

class Detection:
    def __init__(self, options):
        self.prompt = "Here are my docker logs. Your job is to determine if the logs have any anamoly or not. If there is any anamoly, you need to return True, else False. Do not return any other text other than the boolean True or False."
        self.temperature = 0.5
        self.max_tokens = 2048
        self.model_engine = "gpt-3.5-turbo"
        self.client = OpenAI()
        # self.llm = OpenAI(model_name=self.model_engine,
        #                       temperature=self.temperature, max_tokens=self.max_tokens)

    def get_anamoly(self, logs) -> bool:
        # Clean up the logs to remove any sensitive information
        logs = re.sub(r"([0-9a-fA-F]{64})", "<HASH>", logs)
        logs = re.sub(r"([0-9]{1,3}\.){3}[0-9]{1,3}", "<IP_ADDRESS>", logs)

        # Generate a solution using OpenAI's GPT-3 API
        prompt = f"{self.prompt}\n\nLogs:\n{logs}"
        try:
            chat_completion = self.client.chat.completions.create(
            messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
            model="gpt-3.5-turbo",
        )
        except Exception as e:
            raise e
        return (chat_completion.choices[0].message.content.lower() == "true")

if __name__ == "__main__":
    detect = Detection(None)
    print(detect.get_anamoly('''2023-11-05 16:44:27 
2023-11-05 16:44:27 Hello from Docker!
2023-11-05 16:44:27 This message shows that your installation appears to be working correctly.
2023-11-05 16:44:27 
2023-11-05 16:44:27 To generate this message, Docker took the following steps:
2023-11-05 16:44:27  1. The Docker client contacted the Docker daemon.
2023-11-05 16:44:27  2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
2023-11-05 16:44:27     (arm64v8)
2023-11-05 16:44:27  3. The Docker daemon created a new container from that image which runs the
2023-11-05 16:44:27     executable that produces the output you are currently reading.
2023-11-05 16:44:27  4. The Docker daemon streamed that output to the Docker client, which sent it
2023-11-05 16:44:27     to your terminal.
2023-11-05 16:44:27 
2023-11-05 16:44:27 To try something more ambitious, you can run an Ubuntu container with:
2023-11-05 16:44:27  $ docker run -it ubuntu bash
2023-11-05 16:44:27 
2023-11-05 16:44:27 Share images, automate workflows, and more with a free Docker ID:
2023-11-05 16:44:27  https://hub.docker.com/
2023-11-05 16:44:27 
2023-11-05 16:44:27 For more examples and ideas, visit:
2023-11-05 16:44:27  https://docs.docker.com/get-started/
2023-11-05 16:44:27 '''))