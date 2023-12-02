import openai
import posixpath
import argparse

parser = argparse.ArgumentParser("unittest")
parser.add_argument("file", type=argparse.FileType())
parser.add_argument("-n", type=int, default=1)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

rust_file = args.file.read()

openai.api_key_path = posixpath.expanduser("~/.openai")

messages = [
    {
        "role": "user",
        "content": "Here is the source code for a module."
        + "Please generate some useful unit tests for that module in the same language."
        + "Do not write any explanation, only commented rust unit tests:\n"
        + rust_file,
    }
]

response = openai.ChatCompletion.create(model="gpt-4", messages=messages, n=args.n)

print("\n---\n".join([choice.message.content for choice in response.choices]))


if args.verbose:
    print("\n\nCost:")
    print(response.usage)
