import openai
import json
import yfinance as yf
import time

class AssistantManager:
    def __init__(self):
        self.client = openai.OpenAI()

    def run_assistant_and_process(self, content, instructions,
                                  tools_list, function_mapping, model_name="gpt-4-1106-preview"):
        self.function_mapping = function_mapping
        self.assistant = self.client.beta.assistants.create(
            name="Data Analyst Assistant",
            instructions="You are a personal Data Analyst Assistant",
            tools=tools_list,
            model=model_name,
        )

        # Create thread and message as part of the initialization
        self.thread = self.client.beta.threads.create()
        self.message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=content
        )

        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=instructions
        )

        while True:
            time.sleep(1)
            run_status = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=self.run.id)

            if run_status.status == 'completed':
                self.process_completed_run(self.thread)
                break
            elif run_status.status == 'requires_action':
                self.process_required_action(self.thread, run_status, self.run)
            else:
                print("Waiting for the Assistant to process...")

    def process_completed_run(self, thread):
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        for msg in messages.data:
            role = msg.role
            content = msg.content[0].text.value
            print(f"{role.capitalize()}: {content}")

    def process_required_action(self, thread, run_status, run):
        print("Function Calling")
        required_actions = run_status.required_action.submit_tool_outputs.model_dump()
        tool_outputs = []

        for action in required_actions["tool_calls"]:
            func_name = action['function']['name']
            arguments = json.loads(action['function']['arguments'])

            if func_name in self.function_mapping:
                try:
                    output = self.function_mapping[func_name](**arguments)
                    tool_outputs.append({
                        "tool_call_id": action['id'],
                        "output": output
                    })
                except Exception as e:
                    print(f"Error executing {func_name}: {e}")
            else:
                print(f"Unknown function: {func_name}")

        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )


# Assume 'get_stock_price' function is defined elsewhere
def get_stock_price(symbol: str) -> float:
    stock = yf.Ticker(symbol)
    price = stock.history(period="1d")['Close'].iloc[-1]
    return price

# JSON schema and function mapping
tools_list = [{
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Retrieve the latest closing price of a stock using its ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The ticker symbol of the stock"
                }
            },
            "required": ["symbol"]
        }
    }
}]

function_mapping = {"get_stock_price": get_stock_price}

# Initialize AssistantManager with content
assistant_manager = AssistantManager()

# Run assistant and process the result
assistant_manager.run_assistant_and_process(
    "Can you please provide me the stock price of Apple?",
    instructions="Please address the user as Mike",
    tools_list=tools_list,
    function_mapping=function_mapping,
    model_name="gpt-4-1106-preview",
)
