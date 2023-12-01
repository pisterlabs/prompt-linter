# Standard libraries
import argparse
from collections import namedtuple
import logging
import subprocess

# Antheia libraries
from config_manager import ConfigManager
from message_manager import MessageManager
from openai_manager import OpenAIManager
from azure_manager import AzureManager

logger = logging.getLogger(__name__)

ParsedArgs = namedtuple("ParsedArgs", ["audio", "text", "fresh", "low_bandwidth", "costs", "model", "output"])

class Controller:
    def __init__(self, openai_manager=None, azure_manager=None) -> None:
        self.config = ConfigManager.get_instance()
        self.message_manager = MessageManager.get_instance()
        self.openai_manager = openai_manager or OpenAIManager()
        self.azure_manager = azure_manager or AzureManager()

    def run(self) -> None:
        self._show_start_screen()
        self._process_arguments()

    def _show_start_screen(self) -> None:
        print()
        subprocess.run(["imgcat", self.config.CONSOLE_IMAGE])

    def _parse_arguments(self) -> ParsedArgs:
        parser = argparse.ArgumentParser(description="Process arguments.")
        parser.add_argument('--audio', type=str, help='Audio filename for prompt', default=None)
        parser.add_argument('--text', type=str, help='Text for prompt', default=None)
        parser.add_argument('--fresh', action='store_true', help='Starts a new conversation')
        parser.add_argument('--low_bandwidth', action='store_true', help='Optimize audio transfer for low bandwidth')
        parser.add_argument('--costs', action='store_true', help='Displays the API costs')
        parser.add_argument('--model', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'], help="OpenAI model to use.", default='gpt-4')
        parser.add_argument('--output', type=str, choices=['speech', 'text', 'debug'], help="Output to give. Debug gives a dump of the conversation so far.", default="text")
        args = parser.parse_args()
        return ParsedArgs(args.audio, args.text, args.fresh, args.low_bandwidth, args.costs, args.model, args.output)

    def _process_arguments(self) -> None:
        parsed_args = self._parse_arguments()

        self._update_config(parsed_args)
        self._handle_messages(parsed_args)
        self._handle_output(parsed_args)

    def _update_config(self, parsed_args: ParsedArgs) -> None:
        self.config.show_costs = parsed_args.costs
        self.config.low_bandwidth = parsed_args.low_bandwidth
        self.openai_manager.set_model(parsed_args.model)

    def _handle_messages(self, parsed_args: ParsedArgs) -> None:
        if parsed_args.fresh:
            self.message_manager.clear_messages()
        text_prompt = parsed_args.text
        if parsed_args.audio:
            text_prompt = self.antheia.transcribe(parsed_args.audio)
        if not text_prompt:
            text_prompt = input(self.config.colorize('WARNING', 'Enter your message: '))
            print()
        self.message_manager.add_user_message(text_prompt)

    def _handle_output(self, parsed_args: ParsedArgs) -> None:
        output_actions = {
            "speech": self._generate_spoken_response,
            "text": self._generate_written_response,
            "debug": self.config.dump_config
        }
        output_actions[parsed_args.output]()

    def _generate_response(self) -> str:
        response_text = self.openai_manager.generate_response()
        self.message_manager.add_assistant_message(response_text)
        return response_text
    
    def _generate_spoken_response(self) -> None:
        self.azure_manager.synthesize_text_to_speech(self.generate_response())

    def _generate_written_response(self) -> None:
        print(self._generate_response())
        print()


def main() -> None:
    controller = Controller()
    controller.run()

if __name__ == '__main__':
    main()

