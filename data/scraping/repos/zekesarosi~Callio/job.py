import json
import openai
from openai import OpenAI
import csv
import concurrent.futures
import time
import os
import backoff
from log import Log

class Job:
    def __init__(self, config, log_path):
        self.config = config
        self.log = Log(log_path)
        self.api_key = self.config["api_key"]
        self.client = OpenAI(api_key=self.api_key)
        self.input_file_name = self.config["input_file"]
        self.output_file_name = self.config["output_file"]
        self.include_headers = self.config["include_headers"]
        self.keep_data = self.config["keep_data"]
        self.max_workers = self.config["max_workers"]
        self.model = self.config["model"]
        self.input_cost = self.config["input_cost"]
        self.output_cost = self.config["output_cost"]
        self.max_tokens = self.config["max_tokens"]
        self.temperature = self.config["temperature"]
        self.task_timeout = self.config["task_timeout"]
        self.sleep_time = self.config["sleep_time"]
        self.separator = self.config["separator"]

        self.input_headers, self.input_data, self.input_column_data = read_csv_file(self.input_file_name, self.config["input_columns"], self.config["row_start"], self.config["row_end"])
        self.output_column = self.config["output_column"] - 1 
        self.system_msg = self.config["system_msg"]
        self.context = self.config["context"]
        self.output_data = []
        self.cost = 0
        self.message = [{"role": "system", "content": self.system_msg}] + self.context
        self.count = len(self.input_column_data)
        self.input_tokens = 0
        self.output_tokens = 0

    def format_multi_input(self):
        if len(self.input_columns) > 1:
            self.input_column_data = [self.separator.join(map(str, params)) for params in zip(*self.input_column_data)]

    def update_cost(self):
        self.cost = (
            self.input_cost * self.input_tokens / 1000
            + self.output_cost * self.output_tokens / 1000
        )

    def create_workers(self):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            try:
                futures = executor.map(self.response_wrapper, self.input_column_data)
            except Exception as error:
                print(f"Shutting down workers: {error}")
                executor.shutdown(wait=False, cancel_futures=True)
                raise error
        return [future for future in futures]



    def response_wrapper(self, input):
        open_response = response(client=self.client, input=input, model=self.model, context=self.message, timeout=self.task_timeout, sleep_time=self.sleep_time, temperature=self.temperature, max_tokens=self.max_tokens)
        self.input_tokens += open_response.usage.prompt_tokens
        self.output_tokens += open_response.usage.completion_tokens
        self.count -= 1
        self.update_cost()
        print(
            f"Remaining: {self.count} | Cost: ${round(self.cost, 4)} | Input Tokens: {self.input_tokens} | Output Tokens: {self.output_tokens}", end="\r"
        )
        open_response = open_response.choices[0].message.content
        return open_response

    def write_data(self):
        with open(self.output_file_name, "w", newline="") as output_file:
            writer = csv.writer(output_file)
            if self.include_headers:
                writer.writerow(self.input_headers)
            if self.keep_data:
                for i, row in enumerate(self.input_data):
                    try:
                        row[self.output_column] = self.output_data[i]
                    except IndexError:
                        self.log.write("Output column is out of range")
                        for j in range(self.output_column - len(row)):
                            row.append(None)
                        row.append(self.output_data[i])
                writer.writerows(self.input_data)
            else:
                for i in range(len(self.output_data)):
                    writer.writerow([self.output_data[i]])
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Job Complete. Output written to: " + self.output_file_name)
        print("Total Cost: $" + str(round(self.cost, 6)))
        print("Total Tokens: " + str(self.input_tokens + self.output_tokens))
    
    def main(self):
        try:
            self.output_data = self.create_workers()
            self.write_data()
        except Exception as e:
            raise e

def retry_with_exponential_backoff(
    func,
    initial_delay = 1,
    exponential_base = 2,
    jitter = True,
    max_retries = 20,
    errors = (openai.RateLimitError),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except (*errors,) as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception("Max retries exceeded")

                delay *= exponential_base * (1 + (jitter * random.random()))
                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper

def on_giveup(details):
    raise Exception("Max retries exceeded")

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=15)
def response(client, input, model, context, timeout=20, sleep_time=10,temperature=1, max_tokens=500):
    if len(input) > 0:
        messages = context + [{"role": "user", "content": input}]
    else:
        messages = context
    try:
        open_ai_res = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
        )
    except openai.AuthenticationError as error:
        raise error
    except openai.APITimeoutError as error:
        time.sleep(sleep_time)
        open_ai_res = response(client, input, model, context, timeout, sleep_time, temperature, max_tokens)
    except Exception as error:
        raise error
    return open_ai_res 


def format_input(input_columns, separator, input_column_data):
    if len(input_columns) > 1:
        input_column_data = [separator.join(map(str, params)) for params in zip(*input_column_data)]
    else:
        input_column_data = input_column_data[0]
    return input_column_data

def read_csv_file(file_name, input_columns_input, row_start="start", row_end="end", separator=" - ", number=0):
    with open(file_name, "r", newline="") as input_file:
        reader = csv.reader(input_file)
        input_columns = list()
        input_headers = next(reader)
        for column in input_columns_input.split(","):
            input_columns.append(int(column) - 1)
            if int(column) < 1:
                raise Exception("Input column cannot be less than 1")
        missing_columns = set(input_columns) - set(range(len(input_headers)))
        if len(missing_columns) > 0:
            raise Exception(f"Input column(s) {','.join([str(column + 1) for column in missing_columns])} not found in input file")
    
        reader_len = sum(1 for row in reader)
        input_file.seek(0)
        reader = csv.reader(input_file)
        next(reader) # skip headers
        try:
            row_start = (0 if row_start.lower() == "start" else int(row_start))
            if number != 0:
                row_end = row_start + number
            else:
                row_end = (reader_len if row_end.lower() == "end" else int(row_end))
            if row_end > reader_len:
                row_end = reader_len
            if row_start > row_end:
                raise Exception("Row start must be less than row end")
            if row_start < 0:
                raise Exception("Row start cannot be less than 0")
            if row_end < 0:
                raise Exception("Row end cannot be less than 0") 
        except ValueError:
            raise Exception("Row start and row end must be integers or 'start' and 'end'")
            row_start = "start"
            row_end = "end"
        rows = [row for row in reader][row_start:row_end]
        input_data = [row for row in rows if len(row) > 0]
        input_column_data = []
        for column in input_columns:
            input_column_data.append([row[column] for row in input_data])
        input_column_data = format_input(input_columns, separator, input_column_data)
        
        return input_headers, input_data, input_column_data




def main(config, log):
    job = Job(config, log)
    job.main()


if __name__ == "__main__":
    main()
