import argparse
import json
from collections import defaultdict
import time
from openai import OpenAI


def _validate_file(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    print(f'dataset count: {len(dataset)}')

    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

            if not any(message.get("role", None) == "assistant" for message in messages):
                format_errors["example_missing_assistant_message"] += 1

        if format_errors:
            for k, v in format_errors.items():
                print(f"[validate-file] found error: {k}: {v}")
            return False
        return True


def validate_file(args):
    _validate_file(args.file)


def upload_file(args):
    if not _validate_file(args.file):
        print('file validation failed')
        return

    client = OpenAI(api_key=args.token)
    response = client.files.create(
        file=open(args.file, 'rb'), purpose='fine-tune')
    print(f'[upload-file] response: {response}')


def list_files(args):
    client = OpenAI(api_key=args.token)
    response = client.files.list(purpose='fine-tune')
    print(f'[list-files] response: {response}')


def fine_tuning(args):
    client = OpenAI(api_key=args.token)
    response = client.fine_tuning.jobs.create(
        training_file=args.file_id,
        model=args.model,
    )
    print(f'[fine-tuning] response: {response}')

    while True:
        r = client.fine_tuning.jobs.retrieve(response.id)
        if r.status == 'succeeded' or r.status == 'failed' or r.status == 'cancelled':
            print(f'[fine-tuning] status: {r.status}')
            break
        time.sleep(1)


def list_user_models(args):
    client = OpenAI(api_key=args.token)
    response = client.models.list()
    response = [m for m in response if m.owned_by.startswith('user-')]
    print(f'[list-user-models] response: {response}')


def delete_user_model(args):
    client = OpenAI(api_key=args.token)
    response = client.models.delete(args.model)
    print(f'[delete-user-model] response: {response}')


parser = argparse.ArgumentParser(
    prog='python main.py', description='fine-tuning demo')
subparser = parser.add_subparsers(required=True)

parser_validate_file = subparser.add_parser(
    'validate-file', help='validate file')
parser_validate_file.add_argument('--file', type=str,
                                  required=True, help='file path')
parser_validate_file.set_defaults(func=validate_file)

parser_upload_file = subparser.add_parser('upload-file', help='upload file')
parser_upload_file.add_argument('--file', type=str,
                                required=True, help='file path')
parser_upload_file.set_defaults(func=upload_file)

parser_list_files = subparser.add_parser(
    'list-files', help='list uploaded files')
parser_list_files.set_defaults(func=list_files)

parser_fine_tuning = subparser.add_parser(
    'fine-tuning', help='create fining-tune job')
parser_fine_tuning.add_argument('--file_id', type=str,
                                required=True, help='file id')
parser_fine_tuning.add_argument('--model', type=str,
                                help='model (default gpt-3.5-turbo-1106)', default='gpt-3.5-turbo-1106')
parser_fine_tuning.set_defaults(func=fine_tuning)

parser_list_user_models = subparser.add_parser(
    'list-user-models', help='list user models')
parser_list_user_models.set_defaults(func=list_user_models)

parser_delete_user_model = subparser.add_parser(
    'delete-user-model', help='delete user model')
parser_delete_user_model.add_argument('--model', type=str,
                                      help='model to delete', required=True)
parser_delete_user_model.set_defaults(func=delete_user_model)


parser.add_argument('--token', type=str,
                    help='openAI API token', required=True)


def main():
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
