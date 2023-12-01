import os
import argparse
import sys
import tty
import termios
import subprocess
import pathlib

import openai
import pyperclip
import colorama

EXECUTE, COPY, PRINT = "execute", "copy", "print"
CAUTION_FLAGS = {
    "dangerous": (
        "rm",
        "mv",
        "kill",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "dd",
        "wget",
        "curl",
        "-delete",
    ),
    "privileged": ("sudo", "su", "doas", "pkexec", "gksudo", "gksu", "kdesudo", "ksu"),
}


def parse_args(args_override: list = None) -> argparse.Namespace:
    """Parse command line arguments.

    Will exit if the arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Generate a command or chain of shell commands from a natural language prompt. In the default interactive mode, you can cycle through commands with tab, execute with enter or y, copy to the clipboard with c, or quit with q."
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The English-language prompt to use for the GPT completion.",
    )
    execution_mode = (
        parser.add_mutually_exclusive_group()
    )  # Interactive by default, otherwise only one of these can be set
    execution_mode.add_argument(
        "-X",
        f"--{EXECUTE}",
        action="store_true",
        help="Execute mode: runs the generated command immediately without confirmation! Tries to identify dangerous or privileged commands and exit, but should nonetheless be used with great caution.",
    )
    execution_mode.add_argument(
        "-c",
        f"--{COPY}",
        action="store_true",
        help="Copy mode: copies the generated command to the clipboard immediately without prompting for user input.",
    )
    execution_mode.add_argument(
        "-p",
        f"--{PRINT}",
        action="store_true",
        help="Print mode: outputs the generated command directly to stdout without prompting for user input.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Explicitly prefer verbose flags for the generated command, ie. `--help` instead of `-h`.",
    )
    parser.add_argument(
        "-n",
        "--num-prompts",
        type=int,
        default=3,
        metavar=[1, 10],
        help="The maximum number of alternative prompts to generate. Defaults to 3.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0,
        metavar=[0, 2],
        help="Higher values will result in more diverse completions, but lower values will generally result in more sensible ones. Defaults to 0.",
    )
    parser.add_argument(
        "-m",
        "--max-tokens",
        type=int,
        default=100,
        metavar=[1, 1024],
        help="The maximum number of tokens to generate per prompt. Defaults to 100.",
    )
    parser.add_argument(
        "-M",
        "--model",
        type=str,
        default="text-davinci-003",
        help="The GPT model to use. Defaults to text-davinci-003, the latest and most powerful model generally available.",
    )
    parser.add_argument(
        "-K",
        "--api-key",
        type=str,
        default=None,
        help="Manually specify an OpenAI API key to use. If none provided, will default to the OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "-s",
        "--shell",
        type=str,
        default=None,
        help="The shell to use for the generated command, ie. bash, zsh, fish, etc. If none provided, will look for a SHELL environment variable if available, or otherwise default to bash.",
    )
    args = parser.parse_args(args_override)

    if args.temperature < 0 or args.temperature > 2:
        parser.error("Temperature must be between 0.0 and 2.0")
    elif args.num_prompts < 1 or args.num_prompts > 10:
        parser.error("Number of prompts must be between 1 and 10")
    elif args.max_tokens < 1 or args.max_tokens > 1024:
        parser.error("Maximum tokens must be between 1 and 1024")

    if args.execute or args.copy or args.print:
        args.num_prompts = 1  # Save on tokens in any non-interactive mode

    return args


def generate_prompt(args: argparse.Namespace) -> str:
    """Generate the GPT completion prompt from the provided command line arguments

    Determines the user's shell according to the precedence described in the shell argument help text
    If verbose is set, will also prefer verbose flags for the generated command, ie. `--help` instead of `-h`
    """

    shell = None
    if args.shell:
        shell = args.shell
    else:
        env_shell = os.getenv("SHELL")
        if env_shell:
            shell = pathlib.Path(env_shell).name  # ie. /bin/bash -> bash
        else:
            shell = "bash"

    verbosity_clause = (
        " Use verbose flags absolutely wherever possible, ie. --help instead of -h, head --lines=3 instead of n=3, etc.\n"
        if args.verbose
        else "\n"
    )
    sort_clause = (  # Include an additional example to get through GPT's thick skull
        "sort --key=5 --numeric-sort --reverse\nPrompt: Create an annotated git tag for version v1.0\nCommand: git tag --annotate v1.0 --message 'Version 1.0'\n"
        if args.verbose
        else "sort -k5 -n -r\n"
    )
    prompt = (
        f"Convert a prompt into a working programmatic {shell} shell command or chain of commands, making use of standard GNU tools and common Unix idioms."
        + verbosity_clause
        + "Prompt: List all the files in this directory, filtering out the ones that are not directories, and then sort them by size, largest first.\n"
        + "Command: ls -l | grep ^d | "
        + sort_clause
        + f"Prompt: {args.prompt}\n"
        + "Command: "
    )

    return prompt


def get_command_flags(command: str) -> set:
    """Identify dangerous or privileged commands

    Put together this puts reasonable oversight on common commands that could
    potentially delete files, format disks, or run untrusted code
    """

    flags = set()

    for concern in CAUTION_FLAGS:
        for flag in CAUTION_FLAGS[concern]:
            if f"{flag} " in command or command.endswith(flag):
                flags.add(concern)

    return flags


def evaluate_prompt() -> None:
    """Evaluate the prompt and generate a command or chain of commands

    This is the main function that handles the command line arguments, the GPT completion, and the
    interactive command selection loop.
    """
    try:
        args = parse_args()

        if args.api_key:
            openai.api_key = args.api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError(
                "No OpenAI API key provided. Please provide an API key either via the -K/--api-key flag, or by setting OPENAI_API_KEY environment variable in your shell. "
                + "See https://openai.com/docs/developer-quickstart for more information."
            )

        prompt = generate_prompt(args)

        response = openai.Completion.create(
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=1.0,
            n=args.num_prompts,
            frequency_penalty=0.2,
            presence_penalty=0.0,
            stop=["\n"],
        )

        # Deduplicate choices while still preserving ordering (dict is insertion ordered)
        outputs = list(
            {
                choice.text.strip(): get_command_flags(choice.text)
                for choice in response.choices
            }.items()
        )

        if not "".join([output[0] for output in outputs]):
            raise ValueError(
                "No outputs generated. Please try again with a different prompt or model."
            )

        # Queue up the first command for execution or copying
        selected_query = 0
        output, flags = outputs[selected_query]
        flag_string = (
            f" {colorama.Fore.RED}<{', '.join(flags)}>{colorama.Fore.RESET}"
            if flags
            else ""
        )

        # May skip interactive loop in some conditions. Otherwise, enter interactive loop
        if args.execute:
            if flags:
                raise ValueError(
                    f"Command `{output}` triggered cautionary flags{flag_string}\nPlease run in interactive mode to review the command for manual execution."
                )
            action = EXECUTE
        elif args.copy or args.print:
            if flags:
                print(
                    f"Command `{output}` triggered cautionary flags{flag_string}\nPlease review before running!",
                    file=sys.stderr,
                )
            action = COPY if args.copy else PRINT
        else:
            action = None

    except (ValueError, openai.error.OpenAIError) as error:
        sys.exit(f"Error: {error}")

    # Interactive command selection loop
    if not action:
        stdin_descriptor = sys.stdin.fileno()
        stdin_attributes = termios.tcgetattr(stdin_descriptor)
        tty.setraw(stdin_descriptor)

    try:
        while not action:
            print(
                f"({selected_query + 1}/{len(outputs)}{flag_string}): {output}",
                end="\r",
            )
            ch = sys.stdin.read(1)

            if ch == "y" or ch == "\r":
                action = EXECUTE
            elif ch == "c":
                action = COPY
            elif ch == "\t":
                selected_query = (selected_query + 1) % len(outputs)
                output, flags = outputs[selected_query]
            elif (
                ch == "q" or ch == "\x03" or ch == "\x04"
            ):  # SIGKILL & SIGTERM - not ideal tty handling but hey it works
                break

            if not action:
                print("\033[K", end="")  # Clear line
    finally:
        if not args.execute and not args.copy and not args.print:
            termios.tcsetattr(stdin_descriptor, termios.TCSADRAIN, stdin_attributes)

    if action == EXECUTE:
        print(f"Executing: {output}")
        subprocess.run(output, shell=True)
    elif action == COPY:
        print(f"Copying to clipboard: {output}")
        pyperclip.copy(output)
    elif action == PRINT:
        print(output, end="")


if __name__ == "__main__":
    evaluate_prompt()
