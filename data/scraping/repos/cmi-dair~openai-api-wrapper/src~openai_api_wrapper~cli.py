import logging

from openai_api_wrapper import chat_completion, constants, logs, parser

logger = logging.getLogger(logs.LOGGER_NAME)


def main() -> None:
    args = parser.get_parser().parse_args()
    if args.model in constants.CHAT_MODELS:
        response = chat_completion.cli_entrypoint(
            api_key=args.api_key,
            model=args.model,
            messages=args.message,
            messages_file=args.messages_file,
        )
    else:
        raise NotImplementedError(
            f"Model {args.model} is not supported. Supported models are: {constants.SUPPORTED_MODELS}"
        )
    print(response)


if __name__ == "__main__":
    main()
