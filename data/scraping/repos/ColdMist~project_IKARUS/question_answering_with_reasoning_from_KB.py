from langchain.llms import OpenAI
from langchain.tools.json.tool import JsonSpec
from langchain.agents import create_json_agent, AgentExecutor
from langchain.agents.agent_toolkits import JsonToolkit
from utils.helper_functions import *
import argparse

def main(args):
    os.environ["OPENAI_API_KEY"] = args.api_key
    data = read_json_from_file(args.file_path)

    json_spec = JsonSpec(dict_=data, max_value_length=4000)
    json_toolkit = JsonToolkit(spec=json_spec)

    json_agent_executor = create_json_agent(
        llm=OpenAI(temperature=0), toolkit=json_toolkit, verbose=True
    )
    while True:
        print("type your question")
        json_obj = json_agent_executor.run(input())
        #print(json_obj["Thought"][-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a json file.')
    parser.add_argument('--file_path', type=str, help='A json file path.')
    parser.add_argument('--api_key', type=str, help='OpenAI API key.')
    args = parser.parse_args()
    main(args)
