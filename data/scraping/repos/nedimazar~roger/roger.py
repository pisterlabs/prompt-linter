import openai
import json
import os
from pathlib import Path
import sys

from .utils import SETUP, FEWSHOT, METRICS
from .colors import Colors
from .key_manager import get_api_key

USERNAME = "test"


class Roger:
    def __init__(self, buffer_size, instance):
        self.buffer_size = buffer_size
        self.instance = instance
        self.script_dir = Path(__file__).parent.absolute()

        self.validate_instance_directory()
        openai.api_key = get_api_key()
        self.buffer = self.load_buffer()

        self.setup = SETUP
        self.fewshot_prompts = FEWSHOT
        self.metrics = self.load_metrics()

    def validate_instance_directory(self):
        instance_dir = self.script_dir / "memory" / self.instance

        if not instance_dir.exists():
            os.makedirs(instance_dir)

        messages_file = instance_dir / "messages.json"
        metrics_file = instance_dir / "metrics.json"

        if not messages_file.exists():
            with messages_file.open("w") as f:
                json.dump([], f)

        if not metrics_file.exists():
            with metrics_file.open("w") as f:
                json.dump(METRICS, f)

    def load_buffer(self):
        messages_file = self.script_dir / "memory" / self.instance / "messages.json"
        with messages_file.open("r") as f:
            messages = json.load(f)
            return messages[-self.buffer_size :]

    def load_metrics(self):
        metrics_file = self.script_dir / "memory" / self.instance / "metrics.json"
        with metrics_file.open("r") as f:
            metrics = json.load(f)
            return metrics

    def increment_metric(self, metric):
        # Raise an exception if the metrics file or the metric does not exist
        metrics_file = self.script_dir / "memory" / self.instance / "metrics.json"
        with metrics_file.open("r") as f:
            metrics = json.load(f)
            metrics[metric] += 1
        with metrics_file.open("w") as f:
            json.dump(metrics, f)

    def add_message(self, message):
        messages_file = self.script_dir / "memory" / self.instance / "messages.json"
        with messages_file.open("r") as f:
            messages = json.load(f)
            messages.append(message)
        with messages_file.open("w") as f:
            json.dump(messages, f)

        self.buffer = messages[-self.buffer_size :]

    def completion(self, message=None):
        self.increment_metric("invocations")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": self.setup}]
            + self.fewshot_prompts
            + self.buffer
            + [message],
            temperature=0.2,
            top_p=0.1,
        )
        content = response["choices"][0]["message"]["content"]

        return content, response


def wipe_memory(username):
    script_dir = Path(__file__).parent.absolute()
    memory_dir = script_dir / "memory" / username
    # check if there are any backups
    backups = [f for f in memory_dir.iterdir() if f.name.endswith(".bak")]
    # if there  are no backups, renaame the messages file to messages.1.bak
    if len(backups) == 0:
        messages_file = memory_dir / "messages.json"
        messages_file.rename(memory_dir / "messages.1.bak")
    # if there are backups, rename the messages file to messages.n+1.bak
    else:
        n = len(backups)
        messages_file = memory_dir / "messages.json"
        messages_file.rename(memory_dir / f"messages.{n+1}.bak")
    # create a new messages file
    messages_file = memory_dir / "messages.json"
    with messages_file.open("w") as f:
        json.dump([], f)


def log_error(error):
    script_dir = Path(__file__).parent.absolute()
    error_file = script_dir / "errors.log"
    with error_file.open("a") as f:
        f.write(error)
        f.write("\n")


def main():
    # Check if there are command line arguments
    if "--wipe-chat-history-and-backup" in sys.argv:
        wipe_memory(USERNAME)
        # print("wiping memory")
        exit()
    elif len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input(">>> ")

    roger = Roger(15, USERNAME)

    user_message = {"role": "user", "content": user_input}

    # Put this in a try catch block
    try:
        content, response = roger.completion(user_message)
    except Exception as e:
        # Print something went wrong with roger in red
        print(
            Colors.RED
            + "Something went wrong with roger. This is likely an issue with the OpenAI API. You can retry your command and it will probably work. Check errors.log for more information."
            + Colors.RESET
        )
        log_error(str(e))
        exit()

    assistant_message = {"role": "assistant", "content": content}

    roger.add_message(user_message)
    roger.add_message(assistant_message)

    # print roger in bold blue
    print(Colors.BOLD + Colors.BLUE + "roger: " + Colors.RESET, end="")
    print(content)


if __name__ == "__main__":
    main()
