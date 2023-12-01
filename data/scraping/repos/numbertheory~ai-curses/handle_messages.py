import json
from ai_curses import openai


def initialize(args):
    messages = initialize_messages(
        history=args.load_history_json,
        super_command=args.super
    )
    initialize_output(args)
    return messages


def show_meta_help(app):
    add_to_chat_output(
        app,
        "Type \":help\" and then enter or Return"
        " to show additional commands",
        "green_on_black"
    )


def quit_program(messages, args):
    json_dump_text = json.dumps(messages, indent=4)
    if args.output_dir:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            f.write(json_dump_text)
            f.close()
    if args.load_history_json and args.output_md:
        with open(args.output_md, 'a', encoding='utf-8') as f:
            f.write(
                "\n\n## History\n\n"
                "This history was loaded with the `-l` option.\n\n"
                f"```json{json.dumps(messages, indent=4)}```\n\n")
            f.close()
    exit(0)


def process_request(messages, timeout, model):
    return openai.chatgpt(messages, timeout=timeout, model=model)


def add_to_chat_output(app, text, color):
    new_line_split = text.split('\n')
    paragraphs = []
    for line in new_line_split:
        if len(line) < app.cols:
            paragraphs.append(line)
        else:
            chunks, chunk_size = len(line), app.cols
            y = [line[i:i+chunk_size] for i in range(0, chunks, chunk_size)]
            for i in range(0, len(y)):
                paragraphs.append(y[i])
    for para in paragraphs:
        chunks, chunk_size = len(para), app.cols
        scroller = [para[i:i+chunk_size] for i in range(0, chunks, chunk_size)]
        for line in scroller:
            app.print(
                x=0,
                y=app.rows - 5,
                content="{}".format(f"{str(line):<{app.cols}}"),
                panel="layout.0",
                color=color
            )
            app.panels["layout"][0].scroll(1)
            app.screen.refresh()


def initialize_messages(
    history=None,
    super_command="You are a helpful assistant."
):
    if not history:
        return [{"role": "system", "content": f"{super_command}"}]
    else:
        with open(history, 'r') as f:
            messages = json.load(f)

        return messages


def initialize_output(args):
    if args.output_dir:
        with open(args.output_md, 'a', encoding='utf-8') as f:
            f.write(
                "## Prompt\n\n {} \n\n## Conversation\n\n".format(
                    args.super.replace('"""', '')
                )
            )
            f.close()


def message_handler(messages, response, status_code, output_file):
    while len(messages) > 25:
        messages.pop(1)
    if status_code == 200:
        command = messages[-1].get('content')
        messages.append({"role": "assistant", "content": response.strip()})
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write("Human> {} \n\n".format(command))
                f.write("AI> {} \n\n".format(response))
                f.close()
    return messages


def process_helper(messages, app, command, args):
    response, status_code = process_request(
        messages, args.timeout, args.model
    )
    messages = message_handler(
        messages, response, status_code, args.output_md
    )
    add_to_chat_output(
        app, f"Human> {command}", "aqua_on_navy"
    )
    add_to_chat_output(
        app, f"AI> {response}", "black_on_silver"
    )
